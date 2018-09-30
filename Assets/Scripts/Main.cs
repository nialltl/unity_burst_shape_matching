// hello o/
// this is a 2D implementation of the paper Meshless Deformations Based on Shape Matching by Matthias Müller et al, 
// implemented using a Position Based Dynamics integrator and utilising Unity's job system and Burst compiler

// for the original paper refer to https://www.cs.drexel.edu/~david/Classes/Papers/MeshlessDeformations_SIG05.pdf
// for a similar PBD reference implementation (support for small deformations only), see https://github.com/InteractiveComputerGraphics/PositionBasedDynamics/

using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine.Jobs;
using Unity.Mathematics;
using Unity.Burst;
using System.Collections.Generic;

public struct Particle {
    public float2 x; // position
    public float2 p; // predicted position
    public float2 v; // velocity
    public float2 f; // force
    public float inv_mass; // 1.0 / mass. if you want static particles set inv_mass = 0
}

public class Main : MonoBehaviour {
    public NativeArray<Particle> ps;
    int num_particles;
    const int division = 128; // using 128-item sizes jobs for the Job system. I tried various power-of-2's here and 128 seemed to perform well

    // interaction vars
    const int mouse_force = 10;
    const float mouse_influence_radius = 1.0f;

    // shape behaviour
    const float gravity = -9.8f;
    const float stiffness = 0.25f;
    const float linear_deformation_blending = 0.3f; // β in the original paper. extends the range of motion at the cost of looking absolutely wacky

    // shape matching internal state
    const float eps = 1e-6f; // mostly used for simulation safety checks, divide-by-zeroes & static particles
    float dt;
    float2 bounds;
    float2x2 inv_rest_matrix; // shape "rest state" matrix, inverted
    NativeArray<float2> deltas; // precomputed distances of particles to shape center of mass
    NativeArray<float3> com_sums; // temp array for parallelised center of mass summation
    NativeArray<float2x2> shape_matrices; // temp array for parallelised deformed shape matrix summation

    #region Jobs
    #region Integration
    [BurstCompile]
    struct Job_Integrate0 : IJobParallelFor {
        public NativeArray<Particle> ps;

        [ReadOnly] public float dt;

        public void Execute(int i) {
            Particle p = ps[i];
            p.v += p.f * dt;
            p.p += p.v * dt;
            p.f = math.float2(0, gravity);
            ps[i] = p;
        }
    }

    [BurstCompile]
    struct Job_Integrate1 : IJobParallelFor {
        public NativeArray<Particle> ps;

        [ReadOnly] public float2 bounds;
        [ReadOnly] public float inv_dt;

        public void Execute(int i) {
            Particle p = ps[i];

            // Super simple boundary conditions, clamping positions and adding some fake ground friction,
            // by changing the particle's next-frame position to be a little closer to its start position if it penetrated the ground
            if (p.p[1] < -bounds.y) p.p[0] = 0.5f * (p.x[0] + p.p[0]);
            p.p = math.clamp(p.p, -bounds, bounds);
            
            p.v = (p.p - p.x) * inv_dt;
            p.x = p.p;
            ps[i] = p;
        }
    }
    #endregion

    #region Miscellaneous
    [BurstCompile]
    struct Job_ApplyMouseForce : IJobParallelFor {
        public NativeArray<Particle> ps;
        [ReadOnly] public float2 cam_point;
        
        public void Execute(int i) {
            Particle p = ps[i];

            // just calculating a directional force and applying it uniformly to all particles
            float2 dist = (ps[i].p - cam_point);
            p.p += math.normalize(dist) * 0.01f;
            ps[i] = p;
        }
    }
    #endregion

    #region Shape matching jobs
    [BurstCompile]
    struct Job_SumCenterOfMass : IJobParallelFor {
        [WriteOnly] public NativeArray<float3> com_sums;
        [ReadOnly] public NativeArray<Particle> ps;
        [ReadOnly] public int stride;

        public void Execute(int i) {
            // only perform this step from the start of each batch index, every stride'th entry in the array onwards
            if (i % stride != 0) return;
            
            // calculate center of mass for this batch
            float2 cm = math.float2(0);
            
            float wsum = 0.0f;
            for (int idx = i; idx < i + stride; ++idx) {
                Particle p = ps[idx];
                float wi = 1.0f / (p.inv_mass + eps);
                cm += p.p * wi;
                wsum += wi;
            }

            // storing the total weight in the z component for use when combining later
            float3 result = math.float3(cm.x, cm.y, wsum);
            
            com_sums[i] = result;
        }
    }

    [BurstCompile]
    struct Job_SumShapeMatrix : IJobParallelFor {
        [WriteOnly] public NativeArray<float2x2> shape_matrices;
        [ReadOnly] public float2 cm;
        [ReadOnly] public NativeArray<Particle> ps;
        [ReadOnly] public NativeArray<float2> deltas;
        [ReadOnly] public int stride;

        public void Execute(int i) {
            // same idea as in center of mass calculation
            if (i % stride != 0) return;

            // this is part of eq. (7) in the original paper, finding the optimal linear transformation matrix between our reference and deformed positions
            float2x2 mat = math.float2x2(0, 0);
            for (int idx = i; idx < i + stride; ++idx) {
                Particle pi = ps[idx];
                float2 q = deltas[idx];
                float2 p = pi.p - cm;
                float w = 1.0f / (pi.inv_mass + eps);
                p *= w;
                
                mat.c0 += p * q[0];
                mat.c1 += p * q[1];
            }

            shape_matrices[i] = mat;
        }
    }

    [BurstCompile]
    struct Job_GetDeltas : IJobParallelFor {
        public NativeArray<Particle> ps;
        
        [ReadOnly] public float2 cm;
        [ReadOnly] public NativeArray<float2> deltas;
        [ReadOnly] public float2x2 GM;
        
        public void Execute(int i) {
            // calculating the "ideal" position of a particle by multiplying by the deformed shape matrix GM, offset by our center of mass
            float2 goal = math.mul(GM, deltas[i]) + cm;

            // amount to move our particle this timestep for shape matching. if stiffness = 1.0 this corresponds to rigid body behaviour
            // (though you'd need to do more PBD iterations in the main loop to get this stiff enough)
            float2 delta = (goal - ps[i].p) * stiffness;

            Particle p = ps[i];
            p.p += delta;
            ps[i] = p;
        }
    }
    #endregion
    #endregion

    void Start () {
        Screen.SetResolution(500, 500, false);

        dt = Time.fixedDeltaTime;
        bounds = math.float2(4.75f, 4.75f);

        var point_sampler = GameObject.FindObjectOfType<PointSampler>();
        var samples = point_sampler.points;
        var masses = point_sampler.masses;

        // round #samples down to nearest power of 2 if needed, for job system to be able to split workload
        int po2_amnt = 1; while (po2_amnt <= samples.Count) po2_amnt <<= 1;
        num_particles = po2_amnt >> 1;

        ps = new NativeArray<Particle>(num_particles, Allocator.Persistent);

        // populate our array of particles from the samples given, set their initial state
        for (int i = 0; i < num_particles; ++i) {
            float2 sample = samples[i];

            Particle p = new Particle();
            p.x = p.p = math.float2(sample.x, sample.y);
            p.v = p.f = math.float2(0);

            // setting masses based on the greyscale value of our image
            p.inv_mass = 1.0f / masses[i];

            ps[i] = p;
        }

        bool could_init = Init_Body();
        if (!could_init) print("Issue initializing shape");
    }
    
    private bool Init_Body() {
        deltas = new NativeArray<float2>(num_particles, Allocator.Persistent);
        com_sums = new NativeArray<float3>(num_particles, Allocator.Persistent);
        shape_matrices = new NativeArray<float2x2>(num_particles, Allocator.Persistent);
        
        // calculate initial center of mass
        float2 rest_cm = math.float2(0);
        float wsum = 0.0f;
        for (int i = 0; i < num_particles; i++) {
            float wi = 1.0f / (ps[i].inv_mass + eps);
            rest_cm += ps[i].x * wi;
            wsum += wi;
        }
        if (wsum == 0.0) return false;
        rest_cm /= wsum;

        // Calculate inverse rest matrix for use in linear deformation shape matching
        float2x2 A = math.float2x2(0, 0);
        for (int i = 0; i < num_particles; i++) {
            float2 qi = ps[i].x - rest_cm;

            // Caching the position differences for later, they'll never change
            deltas[i] = qi;
            
            // this is forming Aqq, the second term of equation (7) in the paper
            float wi = 1.0f / (ps[i].inv_mass + eps);
            float x2 = wi * qi[0] * qi[0];
            float y2 = wi * qi[1] * qi[1];
            float xy = wi * qi[0] * qi[1];
            A.c0.x += x2; A.c1.x += xy;
            A.c0.y += xy; A.c1.y += y2;
        }
        float det = math.determinant(A);
        if (math.abs(det) > eps) {
            inv_rest_matrix = math.inverse(A);
            return true;
        }
        return false;
    }

    bool Solve_Shape_Matching() {
        JobHandle jh;

        // this stride is used to split the linear summation in both the center of mass, and the calculation of matrix Apq, into parts.
        // these are then calculated in parallel, and finally combined serially
        int stride = ps.Length / division;

        // sum up center of mass in parallel
        var job_sum_center_of_mass = new Job_SumCenterOfMass() {
            ps = ps,
            com_sums = com_sums,
            stride = stride
        };

        jh = job_sum_center_of_mass.Schedule(num_particles, division);
        jh.Complete();

        // after the job is complete, we have the results of each individual summation stored at every "stride'th" array entry of com_sums.
        // the CoM is a float2, but we store the total weight each batch used in the z component. finally we divide the final value by this lump sum
        float2 cm = math.float2(0);
        float sum = 0;
        for (int i = 0; i < com_sums.Length; i += stride) {
            cm.x += com_sums[i].x;
            cm.y += com_sums[i].y;
            sum += com_sums[i].z;
        }
        cm /= sum;

        // calculating Apq in batches, same idea as used for CoM calculation
        var job_sum_shape_matrix = new Job_SumShapeMatrix() {
            cm = cm,
            shape_matrices = shape_matrices,
            ps = ps,
            deltas = deltas,
            stride = stride
        };

        jh = job_sum_shape_matrix.Schedule(num_particles, division);
        jh.Complete();

        // sum up batches and then normalize by total batch count.
        float2x2 Apq = math.float2x2(0, 0, 0, 0);
        for (int i = 0; i < shape_matrices.Length; i += stride) {
            float2x2 shape_mat = shape_matrices[i];
            Apq.c0 += shape_mat.c0;
            Apq.c1 += shape_mat.c1;
        }
        Apq.c0 /= division;
        Apq.c1 /= division;

        // calculating the rotation matrix R, using a 2D polar decomposition.
        // taken from http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
        // this is far more complex in 3D! see e.g. the github PositionBasedDynamics repo for some implementations
        float2 dir = math.float2(Apq.c0.x + Apq.c1.y, Apq.c0.y - Apq.c1.x);
        dir = math.normalize(dir);
        
        float2x2 R = math.float2x2(dir, math.float2(-dir.y, dir.x));
        
        // Calculate A = Apq * Aqq for linear deformations
        float2x2 A = math.mul(Apq, inv_rest_matrix);

        // volume preservation from Müller paper
        float det_A = math.determinant(A);

        // if our determinant is < 0 here, our shape is inverted. if it's 0, it's collapsed entirely.
        if (det_A != 0) {
            // just using the absolute value here for stability in the case of inverted shapes
            float sqrt_det = math.sqrt(math.abs(det_A));
            A.c0 /= sqrt_det;
            A.c1 /= sqrt_det;
        }
         
        // blending between simple shape matched rotation (R term) and the area-preserved deformed shape
        // if linear_deformation_blending = 0, we have "standard" shape matching which only supports small changes from the rest shape. try setting this to 1.0f - pretty amazing
        float2x2 A_term = A * linear_deformation_blending;
        float2x2 R_term = R * (1.0f - linear_deformation_blending);

        // "goal position" matrix composed of a linear blend of A and R
        float2x2 GM = math.float2x2(A_term.c0 + R_term.c0, A_term.c1 + R_term.c1);

        // now actually modify particle positions to apply the shape matching
        var j_get_deltas = new Job_GetDeltas() {
            cm = cm,
            ps = ps,
            deltas = deltas,
            GM = GM
        };
        
        jh = j_get_deltas.Schedule(num_particles, division);
        jh.Complete();
        
        return true;
    }

    public void FixedUpdate() {
        // first integration step applied to all particles
        var j_integrate_0 = new Job_Integrate0() {
            ps = ps,
            dt = dt
        };
        
        JobHandle jh = j_integrate_0.Schedule(num_particles, division);
        jh.Complete();

        // starts solving shape matching constraints here.
        // normally you'd solve any other PBD / XPBD constraints here
        // potentially within a loop, to help constraints converge with a desired stiffness
        Solve_Shape_Matching();

        // mouse interaction
        if (Input.GetMouseButton(0)) {
            // determining mouse position on-screen and passing it to the job
            Vector2 cam_pt = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            float2 cam_point = math.float2(cam_pt.x, cam_pt.y);

            var j_apply_mouse_force = new Job_ApplyMouseForce() {
                ps = ps,
                cam_point = cam_point,
            };

            jh = j_apply_mouse_force.Schedule(num_particles, division);
            jh.Complete();
        }

        // final integration step, including boundary conditions
        var j_integrate_1 = new Job_Integrate1() {
            ps = ps,
            bounds = bounds,
            inv_dt = 1.0f / dt
        };

        jh = j_integrate_1.Schedule(num_particles, division);
        jh.Complete();
    }
    
    private void OnDestroy() {
        ps.Dispose();
        deltas.Dispose();
        com_sums.Dispose();
        shape_matrices.Dispose();
    }
}
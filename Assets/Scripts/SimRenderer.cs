using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using Unity.Mathematics;

public class SimRenderer : MonoBehaviour {
public Mesh instanceMesh;
    [SerializeField] private Material instanceMaterial;

    private Main sim_controller;
    
    // Compute buffers used for indirect mesh drawing
    private ComputeBuffer point_buffer;
    private ComputeBuffer args_buffer;
    private uint[] args = new uint[5] { 0, 0, 0, 0, 0 };
    
    private Bounds bounds;
    
    void Start() {
        sim_controller = GameObject.FindObjectOfType<Main>();

        // Create a compute buffer that holds (# of particles), with a byte offset of (size of particle) for the GPU to process
        point_buffer = new ComputeBuffer(sim_controller.ps.Length, Marshal.SizeOf(new Particle()), ComputeBufferType.Default);
        
        // ensure the material has a reference to this consistent buffer so the shader can access it
        instanceMaterial.SetBuffer("pos_buffer", point_buffer);
        
        // indirect arguments for mesh instances
        args_buffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        uint numIndices = (uint)instanceMesh.GetIndexCount(0);
        args[0] = numIndices;
        args[1] = (uint)sim_controller.ps.Length;
        args_buffer.SetData(args);

        // create rendering bounds for DrawMeshInstancedIndirect
        bounds = new Bounds(Vector3.zero, new Vector3(100, 100, 100));
    }

    void Update() {
        // fill compute buffer with all data of our particles NativeArray, gets passed to our InstancedSimple material
        point_buffer.SetData(sim_controller.ps);

        Graphics.DrawMeshInstancedIndirect(instanceMesh, 0, instanceMaterial, bounds, args_buffer);
    }

    void OnDisable() {
        if (args_buffer != null) args_buffer.Release();
        if (point_buffer != null) point_buffer.Release();
    }

    // Reference -- Slower, but fully CPU-based direct line drawing
    /*public Material mat;
    void OnPostRender() {
        // you should move the creation of circle_offsets out of OnPostRender if you want to use this! mesh can be precomputed
        var circle_offsets = new float2[3];
        const float radius = 0.025f;
        for (int i = 0; i < circle_offsets.Length; ++i) {
            float ang = 2 * Mathf.PI * ((float)i / circle_offsets.Length);
            circle_offsets[i] = new float2(Mathf.Cos(ang) * radius, Mathf.Sin(ang) * radius);
        }

        GL.Begin(GL.LINES);
        mat.SetPass(0);
        GL.Color(Color.white);

        float scale = 0.01f;
        int corr_x = -1024 / 2;
        int corr_y = -768 / 2;
        foreach (var p in sim_controller.ps) {
            float2 x = p.x;
            for (int i = 0; i < circle_offsets.Length; ++i) {
                int next_idx = (i == circle_offsets.Length - 1) ? 0 : i + 1;
                
                float2 pp = math.float2(x.x + corr_x, (-corr_y) - x.y);
                pp *= scale;
                GL.Vertex3(pp.x + circle_offsets[i].x, pp.y + circle_offsets[i].y, 0);
                GL.Vertex3(pp.x + circle_offsets[next_idx].x, pp.y + circle_offsets[next_idx].y, 0);
            }
        }

        GL.End();
    }*/
}
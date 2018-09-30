using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;

public class PointSampler : MonoBehaviour {
    [SerializeField] private Texture2D tex;
    private int tex_width;
    private int tex_height;
    private Color32[] img_colours;

    // number of samples to attempt spawning (many will be rejected)
    const int num_samples = 1024 * 64;
    [HideInInspector] public List<float2> points = new List<float2>();
    [HideInInspector] public List<float> masses = new List<float>();

    void Start() {
        img_colours = tex.GetPixels32();

        tex_width = tex.width;
        tex_height = tex.height;

        // mapping from image dimensions to unity's units - this is just trial + error + preference
        const float scale = 0.025f;
        for (int i = 0; i < num_samples; ++i) {
            float2 p = math.float2(Random.Range(0.0f, tex_width), Random.Range(0.0f, tex_height));

            // reject generated samples if they land in a transparent part of the image
            int idx = QueryIndex((int)p.x, (int)p.y);
            var colour = img_colours[idx];
            if (colour.a < 0.5f) continue;

            // figuring out some mass values based on the greyscale value of the img. 0 = max mass, 1 = minimum
            float brightness = (1.0f / 255) * (colour.r + colour.g + colour.b) / 3;
            brightness = 1.0f - brightness;
            brightness *= brightness;
            masses.Add(math.clamp(brightness * 2, 1.0f, 5.0f));

            // adjusting coordinates to have their origin at the center of the screen
            p.x = (p.x - tex_width / 2.0f) * 0.5f;
            p.y = (p.y - tex_height / 2.0f) * 0.5f;
            p *= scale;

            points.Add(p);
        }
    }
    
    int QueryIndex(int x, int y) {
        // converting x/y image coordinates to unity's GetPixels32 format: 1D array, bottom to top
        return y * tex_width + x;
    }
}

  Shader "Instanced/InstancedShader" {
    Properties {
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _Size ("Size", float) = 0.035
    }

    SubShader {
        Pass {
            Tags { "LightMode"="ForwardBase" "Queue" = "Transparent" "RenderType" = "Transparent" "IgnoreProjector" = "True" }
			ZWrite Off
			Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM

			#pragma glsl
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fwdbase nolightmap nodirlightmap nodynlightmap novertexlight
            #pragma target 4.5
			
            #include "UnityCG.cginc"

			struct Particle {
				float2 x;
				float2 p;
				float2 v;
				float2 f;
				float inv_mass;
			};

			struct v2f {
				float4 pos : SV_POSITION;
				float4 uv : TEXCOORD0;
			};

            float _Size;
            sampler2D _MainTex;
			StructuredBuffer<Particle> pos_buffer;


            v2f vert (appdata_full v, uint instanceID : SV_InstanceID) {
				// Take in data from the compute buffer, filled with data each frame in SimRenderer

                float4 data = float4((pos_buffer[instanceID].p.xy), 0, 1.0);
				
				// Scaling vertices by our base size param (configurable in the material) and the mass of the particle
				float3 localPosition = (v.vertex.xyz * (_Size / pos_buffer[instanceID].inv_mass)) * data.w;
                float3 worldPosition = data.xyz + localPosition;

				// project into camera space
                v2f o;
                o.pos = mul(UNITY_MATRIX_VP, float4(worldPosition, 1.0f));
				o.uv = v.texcoord;

                return o;
            }

            fixed4 frag (v2f i) : SV_Target {
				fixed4 tex_val = tex2D(_MainTex, i.uv);
				
				// example tinting by col_val
				fixed4 col_val = fixed4(1.0, 1.0, 1.0, 1.0);
				fixed4 result = tex_val * col_val;

				return result;
            }

            ENDCG
        }
    }
}
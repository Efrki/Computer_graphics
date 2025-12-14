#version 450

layout (location = 0) in vec3 f_world_pos;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

// глобалки (свет, камера)
layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec3 camera_pos;
	vec3 ambient_light;
};

// материал
layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	vec3 specular_color;
	float shininess;
};

// структуры света
struct DirectionalLight {
	vec3 direction;
	float _pad0;
	vec3 color;
	float intensity;
};

struct PointLight {
	vec3 position;
	float _pad0;
	vec3 color;
	float intensity;
};

struct SpotLight {
	vec3 position;
	float inner_cone;
	vec3 direction;
	float outer_cone;
	vec3 color;
	float intensity;
};

// массивы света
layout (binding = 2, std430) readonly buffer DirectionalLights {
	DirectionalLight dir_lights[];
};

layout (binding = 3, std430) readonly buffer PointLights {
	PointLight point_lights[];
};

layout (binding = 4, std430) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

// текстура
layout (binding = 5) uniform sampler2D u_texture;

void main() {
	vec3 normal = normalize(f_normal);
	
	// сэмплинг
	vec2 distorted_uv = f_uv;
	float time = 0.0; // You can pass time as uniform if needed
	distorted_uv.x += sin(f_uv.y * 10.0 + time) * 0.02;
	distorted_uv.y += cos(f_uv.x * 8.0 + time) * 0.015;
	
	vec3 tex_color1 = texture(u_texture, distorted_uv).rgb;
	vec3 tex_color2 = texture(u_texture, f_uv * 1.2).rgb;
	vec3 tex_color3 = texture(u_texture, f_uv * 0.8).rgb;
	
	vec3 final_tex_color = mix(tex_color1, mix(tex_color2, tex_color3, 0.3), 0.7);
	
	// расчет света
	vec3 color = ambient_light * albedo_color * final_tex_color;

	// направленный свет
	if (dir_lights.length() > 0) {
		vec3 light_dir = normalize(-dir_lights[0].direction);
		float diff = max(dot(normal, light_dir), 0.0);
		color += dir_lights[0].color * dir_lights[0].intensity * diff * albedo_color * final_tex_color;
	}

	// точечный свет
	if (point_lights.length() > 0) {
		vec3 light_dir = point_lights[0].position - f_world_pos;
		float distance = length(light_dir);
		light_dir = normalize(light_dir);
		float attenuation = 1.0 / (distance * distance); // затухание
		float diff = max(dot(normal, light_dir), 0.0);
		color += point_lights[0].color * point_lights[0].intensity * attenuation * diff * albedo_color * final_tex_color;
	}

	// прожектор
	if (spot_lights.length() > 0) {
		vec3 light_dir = spot_lights[0].position - f_world_pos;
		float distance = length(light_dir);
		light_dir = normalize(light_dir);
		float theta = dot(light_dir, normalize(-spot_lights[0].direction)); // конус
		float epsilon = spot_lights[0].inner_cone - spot_lights[0].outer_cone;
		float intensity = clamp((theta - spot_lights[0].outer_cone) / epsilon, 0.0, 1.0);
		float attenuation = 1.0 / (distance * distance);
		float diff = max(dot(normal, light_dir), 0.0);
		color += spot_lights[0].color * spot_lights[0].intensity * attenuation * intensity * diff * albedo_color * final_tex_color;
	}

	final_color = vec4(color, 1.0);
}

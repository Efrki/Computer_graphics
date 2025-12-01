#version 450

layout (location = 0) in vec3 f_world_pos;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec3 camera_pos;
	vec3 ambient_light;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	vec3 specular_color;
	float shininess;
};

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

layout (binding = 2, std430) readonly buffer DirectionalLights {
	DirectionalLight dir_lights[];
};

layout (binding = 3, std430) readonly buffer PointLights {
	PointLight point_lights[];
};

layout (binding = 4, std430) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

void main() {
	vec3 normal = normalize(f_normal);
	vec3 color = ambient_light * albedo_color;

	// Directional light
	if (dir_lights.length() > 0) {
		vec3 light_dir = normalize(-dir_lights[0].direction);
		float diff = max(dot(normal, light_dir), 0.0);
		color += dir_lights[0].color * dir_lights[0].intensity * diff * albedo_color;
	}

	// Point light
	if (point_lights.length() > 0) {
		vec3 light_dir = point_lights[0].position - f_world_pos;
		float distance = length(light_dir);
		light_dir = normalize(light_dir);
		float attenuation = 1.0 / (distance * distance);
		float diff = max(dot(normal, light_dir), 0.0);
		color += point_lights[0].color * point_lights[0].intensity * attenuation * diff * albedo_color;
	}

	// Spot light
	if (spot_lights.length() > 0) {
		vec3 light_dir = spot_lights[0].position - f_world_pos;
		float distance = length(light_dir);
		light_dir = normalize(light_dir);
		float theta = dot(light_dir, normalize(-spot_lights[0].direction));
		float epsilon = spot_lights[0].inner_cone - spot_lights[0].outer_cone;
		float intensity = clamp((theta - spot_lights[0].outer_cone) / epsilon, 0.0, 1.0);
		float attenuation = 1.0 / (distance * distance);
		float diff = max(dot(normal, light_dir), 0.0);
		color += spot_lights[0].color * spot_lights[0].intensity * attenuation * intensity * diff * albedo_color;
	}

	final_color = vec4(color, 1.0);
}

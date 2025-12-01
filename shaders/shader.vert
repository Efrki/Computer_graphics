#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_world_pos;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;

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

void main() {
	vec4 world_pos = model * vec4(v_position, 1.0f);
	mat3 normal_matrix = mat3(transpose(inverse(model)));

	gl_Position = view_projection * world_pos;

	f_world_pos = world_pos.xyz;
	f_normal = normalize(normal_matrix * v_normal);
	f_uv = v_uv;
}

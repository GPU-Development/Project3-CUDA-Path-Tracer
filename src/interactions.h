#pragma once

#include "intersections.h"

#define FRESNEL_EFFECTS 1

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}


/**
* Computes a  random direction for imperfect specular reflection
*/
__host__ __device__
glm::vec3 calculateRandomDirectionReflective(glm::vec3 normal, thrust::default_random_engine &rng, const Material &m) 
{
	thrust::uniform_real_distribution<float> u01(0, 1);

	float n = m.specular.exponent;
	float theta_s = acos(powf(u01(rng),1/(n+1)));
	
	float up = cos(theta_s);
	float over = sin(theta_s);
	float around = u01(rng) * TWO_PI;
	
	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
        Ray &ray,
        glm::vec3 &color,
        const ShadeableIntersection intersect,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	glm::vec3 &l = ray.direction;
	const glm::vec3 &n = intersect.surfaceNormal;


	if (m.hasReflective > 0.0f) {
		ray.origin = getPointOnRay(ray, intersect.t) + .01f * n;
		ray.direction = calculateRandomDirectionReflective(glm::reflect(l, n), rng, m);
		ray.indexOfRefraction = 1.0f;
		color *=  m.specular.color;
	}
	else if (m.hasRefractive > 0.0f) {

		float eta = (intersect.outside) ? 1.0f / m.indexOfRefraction : m.indexOfRefraction;
		float k = 1.0f - eta * eta * (1.0 - glm::dot(-n, l) * glm::dot(-n, l));

		if (k < 0) { //internal reflection
			ray.origin = getPointOnRay(ray, intersect.t) + .01f * n;
			ray.indexOfRefraction = m.indexOfRefraction;
			ray.direction = glm::reflect(l, n);
			color *= m.color;
		}
		else {

			if (intersect.outside) {		
				#if FRESNEL_EFFECTS
				// Schlicks
				float n1 = m.indexOfRefraction;
				float r0 = (1.0f - n1) * (1.0f - n1) / ((1.0f + n1) * (1.0f + n1));
				float r = r0 + (1 - r0) * powf((1 - glm::dot(-n, l)), 5.0f);
				assert(r <= 1.0f);
				thrust::uniform_real_distribution<float> u01(0, 1);

				if (u01(rng) < r) { // reflect
					ray.origin = getPointOnRay(ray, intersect.t) + .01f * n;
					ray.direction = glm::reflect(l, n);
					ray.indexOfRefraction = 1.0f;

					color *=  m.specular.color;
				}
				else 
				#endif
				{				// refract
					ray.origin = getPointOnRay(ray, intersect.t) + .01f * -n;
					ray.indexOfRefraction = m.indexOfRefraction;
					ray.direction = glm::refract(l, n, eta);
					color *= m.color;
				}

			}
			else {
				ray.origin = getPointOnRay(ray, intersect.t) + .01f * -n;
				ray.indexOfRefraction = m.indexOfRefraction;
				ray.direction = glm::refract(l, n, eta);
				color *= m.color;
			}
		}
				
	}
	else {	// assumes diffuse
		ray.origin = getPointOnRay(ray, intersect.t) + .01f * n;
		ray.direction = calculateRandomDirectionInHemisphere(n, rng);
		ray.indexOfRefraction = 1.0f;
		color *= m.color;
	}

}

"""A tiny, dependency-free ray tracer that renders a generic house.

Run with: python examples/ray_tracer.py
The renderer writes examples/house.ppm, an image format supported by most
image viewers and easy to inspect without third-party Python packages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


EPSILON = 1e-5


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vec3":
        return self * (1.0 / scalar)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        return self / self.length()

    def hadamard(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)

    def clamp(self) -> "Vec3":
        return Vec3(*(max(0.0, min(1.0, value)) for value in (self.x, self.y, self.z)))


@dataclass(frozen=True)
class Ray:
    origin: Vec3
    direction: Vec3

    def at(self, distance: float) -> Vec3:
        return self.origin + self.direction * distance


@dataclass(frozen=True)
class Material:
    color: Vec3
    diffuse: float = 0.85
    specular: float = 0.15
    shininess: float = 32.0


@dataclass(frozen=True)
class Hit:
    distance: float
    point: Vec3
    normal: Vec3
    material: Material


@dataclass(frozen=True)
class Triangle:
    a: Vec3
    b: Vec3
    c: Vec3
    material: Material

    def intersect(self, ray: Ray) -> Hit | None:
        edge1, edge2 = self.b - self.a, self.c - self.a
        h = ray.direction.cross(edge2)
        determinant = edge1.dot(h)
        if abs(determinant) < EPSILON:
            return None
        inverse = 1.0 / determinant
        s = ray.origin - self.a
        u = inverse * s.dot(h)
        if u < 0.0 or u > 1.0:
            return None
        q = s.cross(edge1)
        v = inverse * ray.direction.dot(q)
        if v < 0.0 or u + v > 1.0:
            return None
        distance = inverse * edge2.dot(q)
        if distance <= EPSILON:
            return None
        normal = edge1.cross(edge2).normalized()
        if normal.dot(ray.direction) > 0:
            normal = normal * -1
        return Hit(distance, ray.at(distance), normal, self.material)


def quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3, material: Material) -> list[Triangle]:
    return [Triangle(a, b, c, material), Triangle(a, c, d, material)]


def box(low: Vec3, high: Vec3, material: Material) -> list[Triangle]:
    x0, y0, z0, x1, y1, z1 = low.x, low.y, low.z, high.x, high.y, high.z
    p = [Vec3(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    # corners: index = x*4 + y*2 + z
    return (
        quad(p[0], p[1], p[3], p[2], material)  # left
        + quad(p[4], p[6], p[7], p[5], material)  # right
        + quad(p[0], p[4], p[5], p[1], material)  # bottom
        + quad(p[2], p[3], p[7], p[6], material)  # top
        + quad(p[0], p[2], p[6], p[4], material)  # back
        + quad(p[1], p[5], p[7], p[3], material)  # front
    )


def make_scene() -> list[Triangle]:
    ground = Material(Vec3(0.38, 0.52, 0.31), diffuse=0.92, specular=0.02)
    siding = Material(Vec3(0.72, 0.28, 0.14), diffuse=0.9, specular=0.08)
    roof = Material(Vec3(0.20, 0.07, 0.045), diffuse=0.82, specular=0.12)
    trim = Material(Vec3(0.88, 0.78, 0.60), diffuse=0.88, specular=0.1)
    door = Material(Vec3(0.22, 0.10, 0.055), diffuse=0.8, specular=0.18)
    glass = Material(Vec3(0.08, 0.35, 0.52), diffuse=0.55, specular=0.65, shininess=80)
    chimney = Material(Vec3(0.34, 0.11, 0.08), diffuse=0.88, specular=0.05)

    scene = [
        # Ground plane, large enough to catch the cast shadow.
        *quad(Vec3(-20, -0.02, -20), Vec3(20, -0.02, -20), Vec3(20, -0.02, 20), Vec3(-20, -0.02, 20), ground),
        *box(Vec3(-2.8, 0, -1.4), Vec3(2.8, 2.8, 1.4), siding),
        *box(Vec3(-0.75, 0, -1.48), Vec3(0.75, 1.9, -1.51), door),
        *box(Vec3(-2.15, 1.0, -1.5), Vec3(-1.25, 1.95, -1.53), glass),
        *box(Vec3(1.25, 1.0, -1.5), Vec3(2.15, 1.95, -1.53), glass),
        *box(Vec3(-0.10, 0.0, -1.60), Vec3(0.10, 2.0, -1.64), trim),
        *box(Vec3(-2.25, 0.92, -1.60), Vec3(-1.15, 1.04, -1.64), trim),
        *box(Vec3(1.15, 0.92, -1.60), Vec3(2.25, 1.04, -1.64), trim),
        *box(Vec3(-2.1, 0.0, 0.15), Vec3(2.1, 0.75, 1.7), trim),  # front porch
        *box(Vec3(1.25, 2.0, 0.25), Vec3(2.05, 3.8, 0.95), chimney),
    ]

    # Two sloping roof planes meeting at a ridge.
    ridge = Vec3(0, 4.35, 0.0)
    left_front, right_front = Vec3(-3.15, 2.8, -1.65), Vec3(3.15, 2.8, -1.65)
    left_back, right_back = Vec3(-3.15, 2.8, 1.65), Vec3(3.15, 2.8, 1.65)
    scene += quad(left_front, ridge, Vec3(0, 4.35, 1.65), left_back, roof)
    scene += quad(ridge, right_front, right_back, Vec3(0, 4.35, 1.65), roof)
    return scene


def closest_hit(ray: Ray, scene: list[Triangle]) -> Hit | None:
    closest = None
    for triangle in scene:
        hit = triangle.intersect(ray)
        if hit and (closest is None or hit.distance < closest.distance):
            closest = hit
    return closest


def render(width: int, height: int, scene: list[Triangle]) -> list[Vec3]:
    camera = Vec3(8.5, 5.3, -10.5)
    target = Vec3(0, 1.65, 0)
    forward = (target - camera).normalized()
    right = forward.cross(Vec3(0, 1, 0)).normalized()
    up = right.cross(forward).normalized()
    light_direction = Vec3(-0.65, -1.0, -0.45).normalized()  # ray from light to scene
    light_position = Vec3(7, 10, -6)
    pixels: list[Vec3] = []

    for y in range(height):
        for x in range(width):
            aspect = width / height
            screen_x = (2 * ((x + 0.5) / width) - 1) * aspect
            screen_y = 1 - 2 * ((y + 0.5) / height)
            direction = (forward + right * screen_x * 0.72 + up * screen_y * 0.72).normalized()
            ray = Ray(camera, direction)
            hit = closest_hit(ray, scene)
            if hit is None:
                horizon = max(0.0, min(1.0, 0.5 * (direction.y + 0.25)))
                pixels.append(Vec3(0.42, 0.68, 0.90) * (1 - horizon) + Vec3(0.78, 0.88, 1.0) * horizon)
                continue

            to_light = light_position - hit.point
            light_distance = to_light.length()
            shadow_ray = Ray(hit.point + hit.normal * EPSILON * 10, to_light.normalized())
            shadowed = (shadow_hit := closest_hit(shadow_ray, scene)) is not None and shadow_hit.distance < light_distance
            lambert = max(0.0, hit.normal.dot(-1 * light_direction))
            view = (camera - hit.point).normalized()
            reflected = (hit.normal * (2 * hit.normal.dot(-1 * light_direction)) + light_direction).normalized()
            specular = max(0.0, view.dot(reflected)) ** hit.material.shininess
            lighting = 0.22 + (0.12 if shadowed else hit.material.diffuse * lambert)
            color = hit.material.color * lighting + Vec3(1, 1, 1) * (0.18 * hit.material.specular * specular)
            pixels.append(color.clamp())
    return pixels


def write_ppm(path: Path, width: int, height: int, pixels: list[Vec3]) -> None:
    lines = ["P3", f"{width} {height}", "255"]
    lines.extend(" ".join(str(round(channel * 255)) for channel in (pixel.x, pixel.y, pixel.z)) for pixel in pixels)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    width, height = 480, 360
    output = Path(__file__).with_name("house.ppm")
    pixels = render(width, height, make_scene())
    write_ppm(output, width, height, pixels)
    print(f"Rendered {width}x{height} house to {output}")


if __name__ == "__main__":
    main()

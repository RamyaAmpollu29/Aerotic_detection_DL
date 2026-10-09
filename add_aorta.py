import os
import json
import random
import math
from typing import List, Dict

# === Classes from your snippet ===

class Coordinate:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Coordinate(x={self.x}, y={self.y})"

    def to_list(self):
        return [self.x, self.y]

class Region:
    def __init__(self, name: str, coordinates: List[List[int]]):
        self.name = name
        # Convert list of lists into list of Coordinate objects
        self.coordinates = [Coordinate(x, y) for x, y in coordinates]

    def bounding_box(self):
        xs = [c.x for c in self.coordinates]
        ys = [c.y for c in self.coordinates]
        return (min(xs), min(ys), max(xs), max(ys))

    def __len__(self):
        return len(self.coordinates)

    def __repr__(self):
        return f"Region(name={self.name}, points={len(self.coordinates)})"

    def to_list(self):
        """Convert coordinates back to list of lists for JSON serialization"""
        return [c.to_list() for c in self.coordinates]

class RegionsData:
    def __init__(self, data: Dict[str, List[List[int]]]):
        self.regions = {}
        for name, coords in data.items():
            self.regions[name] = Region(name, coords)

    def get_region(self, name: str) -> Region:
        return self.regions.get(name)

    def add_region(self, region: Region):
        self.regions[region.name] = region

    def to_dict(self) -> Dict[str, List[List[int]]]:
        return {name: region.to_list() for name, region in self.regions.items()}

# === Your synthetic Aorta mask generator using Region and Coordinate ===

def generate_aorta_coords(center_x=75, center_y=40, radius=5, num_points=30) -> Region:
    coords = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = int(center_x + radius * random.uniform(0.9, 1.1) * math.cos(angle))
        y = int(center_y + radius * random.uniform(0.9, 1.1) * math.sin(angle))
        coords.append([x, y])
    return Region("Aorta", coords)

# === Main loop: read, add aorta, save ===

input_dir = "masks_json"
output_dir = "masks_json_with_aorta"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, "r") as f:
            mask_data = json.load(f)

        # Wrap existing regions in RegionsData for consistency
        regions_data = RegionsData(mask_data)

        # Generate new Aorta Region and add it
        aorta_region = generate_aorta_coords()
        regions_data.add_region(aorta_region)

        # Save updated JSON with proper conversion
        with open(output_path, "w") as f:
            json.dump(regions_data.to_dict(), f, indent=2)

print("✅ Aorta masks added to all JSON files in:", output_dir)

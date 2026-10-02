import numpy as np
import math
import matplotlib.pyplot as plt
import triangle as tr
import multiprocessing as mp
import queue
import os

class MeshGenerator:
    """
    A class for generating NURBS-based mesh generator for tetral-petal mesh generation
    
    Now uses Approach 1: Single instance that can generate multiple designs
    """

    ATTRIBUTES = ['w1', 'w2', 'w3', 'w4', 'h4', 'd1', 'd2', 'd3']
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # region Initialization and setting/ getting current design parameters
    def __init__(self, boundary_steps=20, half_side=10, add_bounding_box=False):
        """
        Initialize the MeshGenerator with configuration parameters.
        Design parameters are passed to individual generation methods.
        """
        # Configuration parameters (fixed for all designs)
        self.boundary_steps = boundary_steps
        self.half_side = half_side
        self.add_bounding_box = add_bounding_box
        self.cp = np.empty((2, 1))
        
        # Current design parameters (set for each mesh generation)
        self.current_design_params = dict()
        self.successful_meshes = self.successful_meshes = np.empty((0, 9)) # row containing id and design params
    def set_design_parameters(self, *args):
        """Set design parameters for the next mesh generation"""
        for name, value in zip(self.ATTRIBUTES, args):
            self.current_design_params[name] = value

    def set_design_parameters_dict(self, params_dict):
        """Set design parameters from a dictionary"""
        self.current_design_params = params_dict.copy()
    
    def get_current_design_params_as_list(self):
        """Get current design parameters as a list"""
        list(self.current_design_params.values())
    # endregion
    
    # region Utility functions
    @staticmethod
    def make_vector(x, y):
        """Return a 2D vector as a numpy column array."""
        return np.array([[x], [y]])

    @staticmethod
    def deBoor(k: int, x, t, c, p: int):
        """Evaluates S(x) using de Boor's algorithm."""
        d = [c[:, j + k - p] for j in range(0, p + 1)]

        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                denom = (t[j + 1 + k - r] - t[j + k - p])
                alpha = 0 if denom == 0 else (x - t[j + k - p]) / denom
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

        return d[p]

    @staticmethod
    def calculate_B_spline_basis(k: int, x, t, p: int):
        """Calculate B spline basis for point x."""
        d = [1] * (p + 1)

        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                denom = (t[j + 1 + k - r] - t[j + k - p])
                alpha = 0 if denom == 0 else (x - t[j + k - p]) / denom
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

        return d[p]

    def calculate_NURBS_basis(self, k: int, x, t, p: int):
        """Compute normalized NURBS basis function."""
        numerator = self.calculate_B_spline_basis(k, x, t, p)
        basis_sum = numerator + sum(
            self.calculate_B_spline_basis(j, x, t, p) for j in range(k - p, k)
        )
        return numerator / basis_sum
    
    # endregion
    
    # region NURBS curve generation
    def create_NURBS_curves(self, knot, control_points, steps, p):
        """Generate NURBS curve points for given knots and control points."""
        curve_points = []
        for k in range(p, len(knot) - p - 1):
            for x in np.linspace(knot[k], knot[k + 1], steps, endpoint=k == len(knot) - p - 2):
                curve_points.append(self.deBoor(k, x, knot, control_points, p))
        return np.column_stack(curve_points)

    def calculate_boundaries(self, curve_points, control_points, knot, p, thickness):
        """Calculate exterior boundary curves offset by the given thickness."""
        n = curve_points.shape[1]
        exterior_points = []
        i = 0

        numerators = [self.make_vector(0.0, 0.0) for _ in range((p + 1))]
        denominator = [0] * (p + 1)
        local_support_n = p + 1

        for k in range(p, p + 5):
            for x in np.linspace(knot[k], knot[k + 1], self.boundary_steps,
                                 endpoint=len(knot) - p - 1):
                if i + 1 == n:
                    tangent = curve_points[:, i] - curve_points[:, i - 1]
                    dT = np.linalg.norm(curve_points[:, i] - curve_points[:, i - 1])
                else:
                    tangent = curve_points[:, i + 1] - curve_points[:, i]
                    dT = np.linalg.norm(curve_points[:, i + 1] - curve_points[:, i])

                normal = self.make_vector(tangent[1], -tangent[0]) / np.linalg.norm(tangent)

                for j in range(local_support_n):
                    basis = self.calculate_NURBS_basis(k - local_support_n + j, x, knot, p)
                    numerators[j] += basis * dT * normal
                    denominator[j] += basis * dT

                i += 1

            offset_vec = numerators.pop(0) / denominator.pop(0)
            c = control_points[:, k - p: k - p + 1]
            exterior_points.append(c + thickness * offset_vec)
            numerators.append(self.make_vector(0.0, 0.0))
            denominator.append(0)

        # Mirror and append for symmetry
        mirror = self.make_vector(-1, 1)
        for i in range(4, -1, -1):
            exterior_points.append(exterior_points[i] * mirror)

        return np.column_stack(exterior_points)
    # endregion
    
    # region Petal construction 
    def create_petal_curves(self):
        if self.current_design_params is None:
            raise ValueError("No design parameters set. Call set_design_parameters() first.")
        
        # Extract parameters from current design
        w1 = self.current_design_params['w1']
        w2 = self.current_design_params['w2'] 
        w3 = self.current_design_params['w3']
        w4 = self.current_design_params['w4']
        h4 = self.current_design_params['h4']
        d1 = self.current_design_params['d1']
        d2 = self.current_design_params['d2']
        d3 = self.current_design_params['d3']

        symmetry = 4
        angle = math.pi / 2
        rotation_angle = math.pi / 2
        d4 = d3 / (2 * abs(math.cos(angle / 2)))
        h1 = w1 * abs(math.tan(angle / 2))

        snap_direction = self.make_vector(np.cos(angle / 2), np.sin(angle / 2))
        C1 = np.dot(np.array([w1, h1]), snap_direction) * snap_direction
        h2 = (h4 - h1) / 3 + h1
        h3 = 2 * (h4 - h1) / 3 + h1

        C1_normal = np.array([-C1[1], C1[0]]) / np.linalg.norm(C1)
        C2 = C1 + C1_normal * d1

        C3 = self.make_vector(w2, h2)
        C4 = self.make_vector(w3, h3)
        C5 = self.make_vector(w4, h4)

        knot = [0, 0] + list(np.arange(0, 1, 0.125)) + [1] * 3
        mirror = np.array([[-1], [1]])

        cur_angle = rotation_angle / 2
        R = np.array([
            [math.cos(cur_angle), -math.sin(cur_angle)],
            [math.sin(cur_angle), math.cos(cur_angle)]
        ])

        control_points = np.column_stack((
            C1, C2, C3, C4, C5,
            mirror * C5, mirror * C4, mirror * C3,
            mirror * C2, mirror * C1
        ))

        # Generate inner NURBS curve
        interior_points = self.create_NURBS_curves(knot, control_points, self.boundary_steps, 2)

        last_curve_point = self.make_vector(interior_points[0, -1], interior_points[1, -1] + d4)
        lb = self.half_side - abs(R @ last_curve_point)[0, 0] - (d4 if self.add_bounding_box else 0)

        # Exterior boundary
        exterior_control_points = self.calculate_boundaries(interior_points, control_points, knot, 2, d2)

        # Horizontal curve adjustments
        horizontal_curve_points = [interior_points, []]
        cur_points = exterior_control_points
        cur_points[:, 0:1] = (cur_points[:, 0:1].T @ snap_direction) * snap_direction
        cur_points[:, -1:] = (cur_points[:, -1:].T @ (snap_direction * mirror)) * snap_direction * mirror
        horizontal_curve_points[1] = self.create_NURBS_curves(knot, cur_points, self.boundary_steps, 2)

        horizontal_curve_points = np.stack(horizontal_curve_points)
        horizontal_curve_points[:, 1, :] += d4

        # Bar and bounding box assembly
        last_base_points = np.column_stack([
            horizontal_curve_points[0, :, -1:],
            horizontal_curve_points[1, :, -1:]
        ])
        bar_vertical_vec = last_base_points[:, 1:] - last_base_points[:, 0:1]
        bar_vertical_vec /= np.linalg.norm(bar_vertical_vec)
        bar_horizontal_vec = self.make_vector(-bar_vertical_vec[1, 0], bar_vertical_vec[0, 0])
        bar_corner = last_base_points[:, 0:1] + bar_vertical_vec * lb

        vertical_strip = np.column_stack([
            horizontal_curve_points[1, :, -1:],
            (last_base_points[:, 0:1] + bar_corner) / 2, bar_corner
        ])

        vertical_bar_strips = [
            vertical_strip,
            (vertical_strip + d3 * bar_horizontal_vec)[:, ::-1]
        ]

        bar_vertical_points = np.stack(vertical_bar_strips)
        all_points = [horizontal_curve_points, bar_vertical_points]

        # Initialize containers
        inner_vertices, inner_segments = [], []
        outer_vertices, outer_segments = [], []
        bar_end_segments = []

        # Apply symmetry rotations
        for n in range(symmetry):
            cur_angle = rotation_angle * n - rotation_angle / 2
            R = np.array([
                [math.cos(cur_angle), -math.sin(cur_angle)],
                [math.sin(cur_angle), math.cos(cur_angle)]
            ])

            if n == 0:
                self.cp = np.copy(control_points)
                self.cp[1, :] += d4
                self.cp = R @ self.cp

            for k_idx, cur_points in enumerate(all_points):
                for i, new_curve in enumerate(cur_points):
                    new_points = R @ new_curve

                    if i == 0 and k_idx == 0:
                        segments = inner_segments
                        vertices = inner_vertices
                    else:
                        segments = outer_segments
                        vertices = outer_vertices

                    m = len(segments)
                    segments += zip(range(m, m + new_points.shape[1] - 1),
                                    range(m + 1, m + new_points.shape[1]))

                    if k_idx == 0:
                        vertices += new_points.T.tolist()
                    elif i == 0:
                        vertices += new_points.T.tolist()[1:]
                        bar_end_segments.append(segments[-1][1])
                    else:
                        vertices += new_points.T.tolist()[0:1]
                        bar_end_segments.append(segments[-1][0])

                    if i == k_idx == 0:
                        vertices.pop(-1)

                    if n == symmetry - 1 and i == k_idx:
                        segments[-1] = (segments[-1][0], segments[0][0])

        inner_segments = np.array(inner_segments)
        outer_segments = np.array(outer_segments) + inner_segments.shape[0]
    
        all_vertices = np.vstack([inner_vertices, outer_vertices])

        return inner_segments, outer_segments, all_vertices

    def create_petal(self):
        """Create a petal shape using current design parameters."""

        inner_segments, outer_segments, all_vertices = self.create_petal_curves()
        all_segments = np.vstack([inner_segments, outer_segments])

        return (all_segments, all_vertices)
    # endregion
    
    # region Mesh generation
    def generateMesh(self):
        """Generate mesh with pqa0.003 only - fail if not successful"""
        segments, vertices = self.create_petal()
        
        polygon = dict(
            vertices=vertices, 
            segments=segments, 
            holes=np.array([[0, 0]])
        )

        return self.robust_triangulate_pqa0003(polygon)

    @staticmethod
    def triangulate_worker(polygon, result_queue):
        try:
            mesh = tr.triangulate(polygon, 'pqa0.003')
            path = os.path.join(MeshGenerator.BASE_DIR, "temp", "temp_mesh.npz")

            np.savez(path, 
                vertices=mesh['vertices'], 
                triangles=mesh['triangles'])

            result_queue.put(('success', ""))
        except SystemExit as e:
            result_queue.put(('exit', f"Exit code: {e.code}"))
        except MemoryError:
            result_queue.put(('memory_error', "Out of memory"))
        except Exception as e:
            result_queue.put(('error', str(e)))

    def robust_triangulate_pqa0003(self, polygon, timeout_per_design=30): # TODO: adjust timeout threshold if needed
        """Robust triangulation using only pqa0.003 strategy"""
        
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=self.triangulate_worker,
            args=(polygon, result_queue)
        )
        process.daemon = True
        
        try:
            process.start()
            process.join(timeout=timeout_per_design)
            
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                raise TimeoutError(f"Triangulation timed out after {timeout_per_design} seconds")
            
            exit_code = process.exitcode
            if exit_code not in [0, None]:
                error_messages = {
                    3221225477: "Access violation (segfault)",
                    3221225725: "Stack overflow", 
                    3221225786: "Heap corruption",
                    3221225501: "Floating point exception",
                    3221225510: "Illegal instruction",
                    3221225524: "Array bounds exceeded"
                }

                plt.plot(polygon["vertices"][polygon["segments"], 0], polygon["vertices"][polygon["segments"], 1])
                plt.scatter(polygon["holes"][:, 0], polygon["holes"][:, 1])
                plt.show()

                error_msg = error_messages.get(exit_code, f"Exit code: {exit_code}")
                raise RuntimeError(f"Triangulation crashed: {error_msg}")
            
            try:
                status, result = result_queue.get_nowait()
                if status == 'success':
                    path = os.path.join(self.BASE_DIR, "temp", "temp_mesh.npz")
                    data = np.load(path)
                    vertices = data['vertices']
                    triangles = data['triangles']
                    data.close()

                    if len(triangles) == 0:
                        raise RuntimeError("Triangulation produced empty mesh")
                    return vertices, triangles
                elif status == 'exit':
                    raise RuntimeError(f"Process called exit(): {result}")
                elif status == 'memory_error':
                    raise MemoryError("Triangulation out of memory")
                else:
                    raise RuntimeError(f"Triangulation error: {result}")
            except queue.Empty:
                raise RuntimeError("Process finished but no result returned (likely crash)")
                
        except Exception as e:
            raise e

    # endregion
    
    # region Save inp file
    def plotAndSaveMesh(self, nodes, elements, inpFileName, plot=False):
        try:
            targets = {"LEFT": -self.half_side, "RIGHT": self.half_side, "TOP": self.half_side, "BOTTOM": -self.half_side}
            axes = dict(zip(targets.keys(), [[],[],[],[]]))
            
            for index, axis in enumerate(targets):
                target = targets[axis]
                mask = abs(nodes[:, int(index / 2)] - target) <= 1e-4
                b = np.where(mask)[0]

                axes[axis] += [str(a + 1) for a in np.where(mask)[0]]
            
            if plot:
                plt.plot([-10, -10, 10, 10, -10], [10, -10, -10, 10, 10], "red")
                plt.triplot(nodes[:,0], nodes[:,1], elements)
                plt.scatter(self.cp[0, :], self.cp[1, :], color="red")
                plt.axis("equal")
                plt.show()
            
            with open(inpFileName, "w") as file:
                lines = ["*NODE\n"]

                for i in range(len(nodes)):
                    lines.append(f"{i + 1}, {nodes[i][0]}, {nodes[i][1]}, 0\n")

                lines.append("*ELEMENT, TYPE=CPS3\n")

                for i in range(len(elements)):
                    line = f"{i + 1}"
                    for j in range(3):
                        line += f", {elements[i][j] + 1}"
                    lines.append(line + "\n")

                for ax in axes:
                    lines.append(f"*NSET, NSET={ax}\n")

                    i = 0
                    while i < len(axes[ax]):
                        lines.append(", ".join(axes[ax][i:i + 16]) + "\n")
                        i += 16

                lines.append("*ELSET, ELSET=FULLMESH\n")
                element_range = list(str(val) for val in range(1, len(elements) + 1))

                i = 0
                while i < len(elements):
                    lines.append(", ".join(element_range[i:i + 16]) + "\n")
                    i += 16

                lines += [
                    "*MATERIAL, NAME=MATERIAL1\n",
                    "*ELASTIC\n",
                    "3000, 0.35\n",
                    "*SOLID SECTION, ELSET=FULLMESH, MATERIAL=MATERIAL1\n"
                ]

                lines += [
                    "*STEP, NAME=X-AXIS-TENSION, NLGEOM=NO\n",
                    "*STATIC\n",
                    "*OUTPUT, FIELD\n",
                    "*NODE OUTPUT\n",
                    "U\n"
                    "*ELEMENT OUTPUT, DIRECTIONS=YES\n",
                    "IVOL, S\n",
                    "*BOUNDARY\n",
                    "TOP, 2, 2, 0\n",
                    "BOTTOM, 2, 2, 0\n",
                    "LEFT, 1, 1, -5\n",
                    "RIGHT, 1, 1, 5\n",
                    "*ENDSTEP"
                ]

                file.writelines(lines)
        except Exception:
            print("plot or saving mesh failed")
    # endregion
    # region Batch processing
    def generateMeshesBatch(self, design_parameters_list, output_path="batch_results", 
                            plot=False):
        """
        Batch process multiple designs using the same MeshGenerator instance.
        
        Args:
            design_parameters_list: List of parameter tuples or dicts for each design
            output_dir: Directory to save all .inp files
            max_workers: Maximum number of parallel workers
            timeout_per_design: Timeout in seconds for each design
            plot: Whether to generate plots for each design
            
        Returns:
            List of (design_params, result) where result is either (nodes, elements) or Exception
        """
        from pathlib import Path
        import time
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        def process_single_design(design_params, design_id):
            """Process a single design with the shared MeshGenerator instance"""
            start_time = time.time()
            try:
                # Set design parameters for this specific design
                if isinstance(design_params, dict):
                    self.set_design_parameters_dict(design_params)
                else:
                    self.set_design_parameters(*list(design_params))
                
                # Generate mesh using the robust method
                nodes, elements = self.generateMesh()
                
                # Save to file
                job_name = f"job{design_id}"
                job_dir = output_dir / job_name # each job has own directory
                job_dir.mkdir(parents = True, exist_ok=True)
                inp_filename = job_dir / f"{job_name}.inp"

                self.plotAndSaveMesh(nodes, elements, str(inp_filename), plot=plot)
                elapsed = time.time() - start_time
                print(f"Design {design_id} completed in {elapsed:.2f}s - {len(elements)} triangles")
                return (design_params, (nodes, elements))
                
            except Exception as e:
                elapsed = time.time() - start_time
                error_msg = f"Design {design_id} failed after {elapsed:.2f}s: {str(e)}"
                print(error_msg)
                return (design_params, e)
    
        # Process all sets of params sequentially
        results = []
        completed = 0
        successful = 0

        total_designs = len(design_parameters_list)
        
        print(f"Starting batch processing of {total_designs} designs...")
        print(f"Output directory: {output_dir.absolute()}")
        
        for idx, design_params in enumerate(design_parameters_list):
            design_id = f"design_{idx:04d}"  # 4 decimals
            try:
                result = process_single_design(design_params, design_id)
                results.append(result)
                
                completed += 1
                if not isinstance(result[1], Exception):
                    successful += 1
                #self.successful_meshes = np.vstack([self.successful_meshes, np.concatenate([[design_id], design_params])]) # add row containing id and params
                print(f"Progress: {completed}/{total_designs} ({completed/total_designs*100:.1f}%) Success count: {successful}")
            
            except Exception as e:
                error_msg = f"Unexpected error processing design {design_id}: {str(e)}"
                print(error_msg)
                results.append((design_params, RuntimeError(error_msg)))
                completed += 1
        
        # Print summary
        success_count = sum(1 for r in results if not isinstance(r[1], Exception))
        failure_count = len(results) - success_count
        
        print(f"\nBatch processing completed:")
        print(f"  Successful: {success_count}/{total_designs}")
        print(f"  Failed: {failure_count}/{total_designs}")
        print(f"  Success rate: {success_count/total_designs*100:.1f}% Success count : {success_count}")
        
        return results
    
    # endregion


# TEST

if __name__ == '__main__':
    mesh_generator = MeshGenerator()
    results = mesh_generator.generateMeshesBatch(np.array([[1.0898825900854086,1.7365447452994442,2.944910031806847,2.949297752057866,8.68843293462024,0.8731533845340957,0.9261324001578355,0.2578740481762315]]))
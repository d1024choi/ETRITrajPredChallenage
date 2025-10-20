from libraries import *

class ProcessMap:
    def __init__(self):
       self.dim = 2

    def __call__(self, lane_segments):

        lane_segments = lane_segments
        num_ids = self.get_number_of_ids(lane_segments)
        num_polygons = num_ids
        # initialization
        polygon_position = torch.zeros(num_polygons, self.dim, dtype=torch.float)
        polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
        start_node_map = {}
        end_node_map = {}
        polygon_to_polygon_edge_index = []

        """
            polygon_to_polygon_edge_index is a list of connections between lane_segments.

                “If I start where someone ended → I connect to them”

                “If I end where someone starts → I connect to them"
                
                A----------B---------C--------D
                
                A is connected to B
                B is connected to  C and A 
                C is connected to B and D
        
        """

        for idx, lane_segment in enumerate(lane_segments):
            lane_id = lane_segment['ID']
            speed_limit = lane_segment['Speed']
            start_node = lane_segment['SNodeID']
            end_node = lane_segment['ENodeID']
            points = lane_segment['Pts']  # (NumPts, 2 or 3)
            orientation = lane_segment['Ori']    # We can utilize this Orientation as well
            bounding_box = lane_segment['Cover']

            centerline = points[:, :self.dim]
            # Convert to torch tensor
            centerline = torch.from_numpy(centerline).float()
            polygon_position[idx] = centerline[0, :self.dim]
            polygon_orientation[idx] = torch.atan2(centerline[1, 1] - centerline[0, 1],
                                                                centerline[1, 0] - centerline[0, 0])


            polygon_type[idx] = lane_segment['Type']

            point_position[idx] = centerline[:-1, :self.dim]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[idx] = torch.atan2(center_vectors[:, 1], center_vectors[:, 0])
            point_magnitude[idx] =  torch.norm(center_vectors[:, :2], p=2, dim=-1)

            if end_node in start_node_map:
                for neighbor_idx in start_node_map[end_node]:
                    polygon_to_polygon_edge_index.append((idx, neighbor_idx))

                # Backward connections: segments ending at current's start node to current segment
            if start_node in end_node_map:
                for neighbor_idx in end_node_map[start_node]:
                    polygon_to_polygon_edge_index.append((neighbor_idx, idx))

                # Add current segment to maps for future connections
            if start_node not in start_node_map:
                start_node_map[start_node] = []
            start_node_map[start_node].append(idx)

            if end_node not in end_node_map:
                end_node_map[end_node] = []
            end_node_map[end_node].append(idx)

        if polygon_to_polygon_edge_index:
            polygon_to_polygon_edge_index = torch.tensor(polygon_to_polygon_edge_index, dtype=torch.long).t()
        else:
            polygon_to_polygon_edge_index = torch.empty((2, 0), dtype=torch.long)
        num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
        point_to_polygon_edge_index = torch.stack(
                [torch.arange(num_points.sum(), dtype=torch.long),
                 torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)

        map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
        }
        map_data['map_polygon']['num_nodes'] = num_polygons
        map_data['map_polygon']['position'] = polygon_position
        map_data['map_polygon']['orientation'] = polygon_orientation
        map_data['map_polygon']['type'] = polygon_type

        if len(num_points) == 0:
            map_data['map_point']['num_nodes'] = 0
            map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)


        else:
            map_data['map_point']['num_nodes'] = num_points.sum().item()
            map_data['map_point']['position'] = torch.cat(point_position, dim=0)
            map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0)
            map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)


        map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index

        return map_data


    def get_number_of_ids(self, lane_segments):
        unique_ids = {segment['ID'] for segment in lane_segments}
        return len(unique_ids)









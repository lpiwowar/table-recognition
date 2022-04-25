import os
import tempfile

import cv2
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm

from table_recognition.graph import Graph
from table_recognition.models import SimpleModel
from table_recognition.models import NodeEdgeMLPEnding
from table_recognition.models import VisualNodeEdgeMLPEnding


class Infer(object):
    def __init__(self, config):
        self.config = config
        self.prepared_data_dir = tempfile.mkdtemp()
        self.visualize_path = tempfile.mkdtemp()

        self.available_models = {
            SimpleModel.__name__: SimpleModel,
            NodeEdgeMLPEnding.__name__: NodeEdgeMLPEnding,
            VisualNodeEdgeMLPEnding.__name__: VisualNodeEdgeMLPEnding
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.available_models[self.config.model_name]()
        self.model.load_state_dict(torch.load(self.config.weights_path, map_location=torch.device(self.device)))
        self.model.eval()

        self.model_nodes = self.available_models[self.config.model_name]()
        self.model_nodes.load_state_dict(torch.load(self.config.weights_path_nodes,
                                                    map_location=torch.device(self.device)))
        self.model_nodes.eval()

        self.prepare_input()

    def prepare_input(self):
        images = os.listdir(self.config.img_path)
        images.sort()
        ocr_output_path = os.listdir(self.config.ocr_output_path)
        ocr_output_path.sort()
        self.config.prepared_data_dir = self.prepared_data_dir
        self.config.visualize_dir = self.visualize_path

        self.config.logger.info(f"Storing prepared graph representations to {self.prepared_data_dir}")
        self.config.logger.info(f"Visualization of output graph stored in {self.visualize_path}")

        counter = 0
        # idx 35: ctdar_439
        # idx 5: ctdar_047
        # idx 21: simple table
        # idx 61: super simple 759?
        # idx 77 : cTDaR_913
        # idx 43 : cTDaR_560
        img_id = 21
        for image_name, ocr_output in tqdm(zip(images[img_id:img_id+1], ocr_output_path[img_id:img_id+1])):
        # for image_name, ocr_output in tqdm(zip(images, ocr_output_path)):
            # print(f"{image_name} {counter}")
            # counter += 1
            # continue
            self.config.logger.info("Preparing graph representation of the input tables ...")
            graph = Graph(
                config=self.config,
                ocr_output_path=os.path.join(self.config.ocr_output_path, ocr_output),
                ground_truth_path=None,
                img_path=os.path.join(self.config.img_path, image_name)
            )

            graph.initialize()
            graph.color_input()

            graph.dump_infer()

            name = image_name.split(".")[0]
            file_path = os.path.join(self.prepared_data_dir, name) + ".pt"
            data = torch.load(os.path.join(file_path))

            self.config.logger.info(f"Running image {data.img_path} through the model ...")
            _, out_edges = self.model(data)
            out_nodes, _ = self.model_nodes(data)

            self.config.logger.info(f"Creating table representation of the table in {data.img_path} image ...")
            out_nodes = torch.argmax(torch.exp(out_nodes), dim=1)
            out_edges = torch.argmax(torch.exp(out_edges), dim=1)

            id_to_node = {node.id: node for node in graph.nodes}
            id_to_edge = {(edge.node1.id, edge.node2.id): edge for edge in graph.edges}

            node_num_to_name = {0: "header", 1: "data"}
            for id1, node_type in enumerate(out_nodes):
                node_type = int(node_type.numpy())
                id_to_node[id1].type = node_num_to_name[node_type]

            # TODO: Switch "horizontal" and "vertical" after get_edge_type() is fixed in output.py
            edge_num_to_name = {0: "cell", 1: "vertical", 2: "horizontal", 3: "no-relationship"}
            for id1, id2, edge_type in zip(data.edge_index[0], data.edge_index[1], out_edges):
                id1 = int(id1.numpy())
                id2 = int(id2.numpy())
                edge_type = int(edge_type.numpy())
                id_to_edge[(id1, id2)].type = edge_num_to_name[edge_type]

            # -- Remove no-relationship edges and run again ------------------------------------------------------------
            """
            graph.edges = [edge for edge in graph.edges if edge.type != "no-relationship"]
            Graph2Table(graph, self.config, self.visualize_path, solo_cleanup=True)

            data_edge_index = data.edge_index.T
            edge_ids_to_keep = []
            for edge_id, edge_index in enumerate(data_edge_index):
                # is_not_no_relationship = [edge for edge in graph.edges if edge.connects(edge_index[0], edge_index[1])]
                is_not_no_relationship = [edge for edge in graph.edges if edge.node1.id == edge_index[0] and edge.node2.id == edge_index[1]]
                if is_not_no_relationship:
                    edge_ids_to_keep += [edge_id]

            data.edge_index = data_edge_index[edge_ids_to_keep].T
            data.edge_attr = data.edge_attr[edge_ids_to_keep]
            data.edge_image_regions = data.edge_image_regions[edge_ids_to_keep]

            _, out_edges = self.model(data)
            out_edges = torch.argmax(torch.exp(out_edges), dim=1)
            id_to_edge = {(edge.node1.id, edge.node2.id): edge for edge in graph.edges}
            for id1, id2, edge_type in zip(data.edge_index[0], data.edge_index[1], out_edges):
                id1 = int(id1.numpy())
                id2 = int(id2.numpy())
                edge_type = int(edge_type.numpy())
                id_to_edge[(id1, id2)].type = edge_num_to_name[edge_type]
            """
            # ----------------------------------------------------------------------------------------------------------
            graph.visualize(img_destination=os.path.join(self.visualize_path, "before_" + image_name))

            Graph2Table(graph, self.config, self.visualize_path, image_name)


class Graph2Table(object):
    def __init__(self, graph, config, visualize_path, image_name, solo_cleanup=False):
        self.config = config
        self.graph = graph
        self.visualize_path = visualize_path
        self.nodes_edges = {}

        # -- 1) Graph cleanup ---------------------------------------------------------------------------
        self.remove_no_relationship()
        self.merge_cell()
        self.remove_symmetrical_edges()
        self.remove_transitive_edges()

        if solo_cleanup:
            return
        self.remove_isolated_nodes()
        # -----------------------------------------------------------------------------------------------

        # -- 2) Straighten edges ------------------------------------------------------------------------
        self.graph.edges = list(set(self.graph.edges))
        self.lock_edges()
        self.vertical_edges = [edge for edge in self.graph.edges
                               if edge.type == "vertical" and not edge.gridify_locked]
        self.horizontal_edges = [edge for edge in self.graph.edges
                                 if edge.type == "horizontal" and not edge.gridify_locked]

        self.horizontal_cc = []
        while self.horizontal_edges:
            new_horizontal_cc = []
            self.get_continuous_component(self.horizontal_edges[0].node2,
                                          self.horizontal_edges,
                                          new_horizontal_cc)
            self.horizontal_cc += [new_horizontal_cc]

        for horizontal_cc in self.horizontal_cc:
            horizontal_cc.sort(key=lambda item: item.grid_y)
            for node in horizontal_cc:
                node.grid_y = horizontal_cc[0].grid_y

        self.vertical_cc = []
        while self.vertical_edges:
            new_vertical_cc = []
            self.get_continuous_component(self.vertical_edges[0].node2,
                                          self.vertical_edges,
                                          new_vertical_cc)
            self.vertical_cc += [new_vertical_cc]

        for vertical_cc in self.vertical_cc:
            vertical_cc.sort(key=lambda item: item.grid_x)
            for node in vertical_cc:
                node.grid_x = vertical_cc[0].grid_x
        # -----------------------------------------------------------------------------------------------

        # -- 3) "Gridify" nodes ---------------------------------------------------------------------------
        self.nodeid_to_edge = {}
        for edge in self.graph.edges:
            self.nodeid_to_edge[edge.node1.id] = self.nodeid_to_edge.get(edge.node1.id, [])
            self.nodeid_to_edge[edge.node1.id] += [edge]
            self.nodeid_to_edge[edge.node2.id] = self.nodeid_to_edge.get(edge.node2.id, [])
            self.nodeid_to_edge[edge.node2.id] += [edge]

        # 3.1) Get list of rows
        self.horizontal_edges = [edge for edge in self.graph.edges
                                 if edge.type == "horizontal" and not edge.gridify_locked]
        self.horizontal_ccs = []
        while self.horizontal_edges:
            new_horizontal_cc_nodes = []
            self.get_continuous_component(self.horizontal_edges[0].node2,
                                          self.horizontal_edges,
                                          new_horizontal_cc_nodes)
            new_row_object = HorizontalCC(new_horizontal_cc_nodes)
            self.horizontal_ccs += [new_row_object]

        horizontally_isolated_nodes = []
        for node in self.graph.nodes:
            if all([edge.gridify_locked for edge in self.nodeid_to_edge[node.id] if edge.type == "horizontal"]):
                horizontally_isolated_nodes += [node]
        self.horizontal_ccs += [HorizontalCC([node]) for node in horizontally_isolated_nodes]
        print(self.horizontal_ccs)

        # for horizontal_cc in self.horizontal_ccs:
        #     horizontal_cc.expand_bbox(self.nodeid_to_edge)

        # 3.2) Get list of columns
        self.vertical_edges = [edge for edge in self.graph.edges
                               if edge.type == "vertical" and not edge.gridify_locked]
        self.vertical_ccs = []
        while self.vertical_edges:
            new_vertical_cc_nodes = []
            self.get_continuous_component(self.vertical_edges[0].node2,
                                          self.vertical_edges,
                                          new_vertical_cc_nodes)
            self.vertical_ccs += [VerticalCC(new_vertical_cc_nodes)]

        vertically_isolated_nodes = []
        for node in self.graph.nodes:
            if all([edge.gridify_locked for edge in self.nodeid_to_edge[node.id] if edge.type == "vertical"]):
                vertically_isolated_nodes += [node]
        self.vertical_ccs += [VerticalCC([node]) for node in vertically_isolated_nodes]
        print(self.vertical_ccs)
        # for vertical_cc in self.vertical_ccs:
        #     vertical_cc.expand_bbox(self.nodeid_to_edge)

        # 3.3) Let columns fall
        self.fall_vertical_cc()

        # 3.4) Let rows fall
        self.fall_horizontal_cc()
        # -----------------------------------------------------------------------------------------------

        # -- Transform "gridified" graph to table ---------------------------------------------------------
        graph.visualize(img_destination=os.path.join(self.visualize_path, "after_" + image_name))
        table_grid = self.generate_table_grid(self.nodeid_to_edge)
        self.generate_html(table_grid)
        # -----------------------------------------------------------------------------------------------

    def generate_html(self, table_grid):
        # row_to_nodes = {}
        # for node in self.graph.nodes:
        #     row_to_nodes[node.grid_row] = row_to_nodes.get(node.grid_row, [])
        #     row_to_nodes[node.grid_row] += [node]
        nodeid_to_node = {node.id: node for node in self.graph.nodes}

        output_html = " <!DOCTYPE html>\n"
        output_html += "<html>\n"
        output_html += "    <head>\n"
        output_html += "        <style>\n"

        output_html += "            table\n"
        output_html += "            {\n"
        output_html += "                border: 3px solid black;\n"
        output_html += "            }\n"

        output_html += "            th\n"
        output_html += "            {\n"
        output_html += "                border: 3px solid black;\n"
        output_html += "                text-align: center;"
        output_html += "            }\n"

        output_html += "            td\n"
        output_html += "            {\n"
        output_html += "                border: 3px solid black;\n"
        output_html += "                background-color: rgb(213,232,212);"
        output_html += "                text-align: center;"
        output_html += "            }\n"
        output_html += "            td.header { background-color: rgb(255,230,204); }"
        output_html += "            td.data { background-color: rgb(213,232,212); }"
        output_html += "            td.empty { background-color: rgb(255,255,255); }"

        output_html += "        </style>\n"
        output_html += "    </head>\n"
        output_html += "    <body>\n"
        output_html += "       <table>\n"

        img = cv2.imread(self.graph.img_path)
        visualize_img_name = "graph_" + self.graph.img_path.split("/")[-1]
        img_name_prefix = self.graph.img_path.split("/")[-1].split(".")[0]
        output_dir_path = os.path.join(self.config.html_output_dir, img_name_prefix)
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        for row_idx in range(table_grid.shape[0]):
            output_html += "<tr>\n"

            for column_idx in range(table_grid.shape[1]):
                current_value = table_grid[row_idx, column_idx]
                if current_value != -1 and current_value != -2:
                    [(min_x, min_y), (max_x, max_y)] = nodeid_to_node[current_value].grid_bbox
                    [(min_x, min_y), (max_x, max_y)] = [(max(0, min_x), max(0, min_y)), (max(0, max_x), max(0, max_y))]
                    img_region = img[min_y:max_y, min_x:max_x]
                    img_path = os.path.join(output_dir_path, f"{current_value}.jpg")
                    cv2.imwrite(img_path, img_region)

                    colspan = nodeid_to_node[current_value].grid_colspan
                    rowspan = nodeid_to_node[current_value].grid_rowspan

                    node_img_name = f"{current_value}.jpg"
                    node_type = nodeid_to_node[current_value].type
                    output_html += f'<td class="{node_type}" colspan={colspan} rowspan={rowspan}>\n'
                    output_html += f'<img src={node_img_name}>\n'
                    output_html += f"</td>\n"
                    table_grid[table_grid == current_value] = -2
                elif current_value != -2:
                    output_html += f'<td class="empty">Empty</td>\n'
                else:
                    continue

            output_html += "</tr>\n"

        import shutil

        shutil.copyfile(os.path.join(self.visualize_path, "before_" + self.graph.img_path.split("/")[-1]),
                        os.path.join(output_dir_path, "before_" + self.graph.img_path.split("/")[-1]))
        shutil.copyfile(os.path.join(self.visualize_path, "after_" + self.graph.img_path.split("/")[-1]),
                        os.path.join(output_dir_path, "after_" + self.graph.img_path.split("/")[-1]))
        shutil.copyfile(os.path.abspath(self.graph.img_path),
                        os.path.join(output_dir_path, self.graph.img_path.split("/")[-1]))

        output_html += "        </table>\n"
        # output_html += f'            <img src={os.path.abspath(os.path.join(output_dir_path, "after_" + visualize_img_name))}>'
        # output_html += f'            <img src={os.path.abspath(os.path.join(output_dir_path, "before_" + visualize_img_name))}>'
        # output_html += f'            <img src={os.path.abspath(os.path.join(output_dir_path, visualize_img_name))}>'

        src = "./after_" + self.graph.img_path.split("/")[-1]
        output_html += f'            <img src={src}>'

        src = "./before_" + self.graph.img_path.split("/")[-1]
        output_html += f'            <img src={src}>'

        src = "./" + self.graph.img_path.split("/")[-1]
        output_html += f'            <img src={src}>'

        # output_html += f"            <img src={os.path.abspath(os.path.join(self.visualize_path, visualize_img_name))}>"
        # output_html += f"            <img src={os.path.abspath(os.path.abspath(self.graph.img_path))}>"
        output_html += "    </body>\n"
        output_html += "</html>\n"

        output_html_path = os.path.join(output_dir_path, f"{img_name_prefix}.html")

        with open(output_html_path, "w") as f:
            f.write(output_html)

    def generate_table_grid(self, nodeid_to_edge):
        max_num_rows = max([node.grid_row for node in self.graph.nodes]) + 1
        max_num_columns = max([node.grid_column for node in self.graph.nodes]) + 1
        table_grid = np.full((max_num_rows, max_num_columns), -1)

        # Calculate nodes col and row span
        """
        for node in self.graph.nodes:
            nodes_edges = nodeid_to_edge[node.id]
            nodes_vertical_neighbours = [edge for edge in nodes_edges if edge.type == "vertical"]
            nodes_horizontal_neighbours = [edge for edge in nodes_edges if edge.type == "horizontal"]
            print(nodes_horizontal_neighbours)
            print(nodes_vertical_neighbours)
            if nodes_vertical_neighbours:
                max_grid_x = max([edge.get_connected_node(node.id).grid_x for edge in nodes_vertical_neighbours])
                min_grid_x = min([edge.get_connected_node(node.id).grid_x for edge in nodes_vertical_neighbours])

                # node.grid_x = min_grid_x
                node.colspan = (max_grid_x - min_grid_x) + 1

            if nodes_horizontal_neighbours:
                max_grid_y = max([edge.get_connected_node(node.id).grid_y for edge in nodes_horizontal_neighbours])
                min_grid_y = min([edge.get_connected_node(node.id).grid_y for edge in nodes_horizontal_neighbours])

                # node.grid_y = min_grid_y
                node.rowspan = (max_grid_y - min_grid_y) + 1

            print(f"node_id: {node.id} grid_x: {node.grid_x} grid_y: {node.grid_y} colspan: {node.grid_colspan} rowspan: {node.grid_rowspan}")
        """
        """
        for node in self.graph.nodes:
            nodes_edges = nodeid_to_edge[node.id]

            nodes_edges_horizontal = [edge for edge in nodes_edges if edge.type == "horizontal"]
            nodes_edges_horizontal_right = [edge for edge in nodes_edges_horizontal
                                            if edge.get_connected_node(node.id).grid_x > node.grid_x]

            nodes_edges_vertical = [edge for edge in nodes_edges if edge.type == "vertical"]
            nodes_edges_vertical_below = [edge for edge in nodes_edges_vertical
                                          if edge.get_connected_node(node.id).grid_y > node.grid_y]

            new_colspan = None
            if nodes_edges_vertical_below:
                nodes_edges_vertical_below_max_grid_column = max([edge.get_connected_node(node.id).grid_column
                                                                  for edge in nodes_edges_vertical_below])
                nodes_edges_vertical_below_min_grid_column = min([edge.get_connected_node(node.id).grid_column
                                                                  for edge in nodes_edges_vertical_below])
                new_colspan = nodes_edges_vertical_below_max_grid_column - nodes_edges_vertical_below_min_grid_column + 1

            new_rowspan = None
            if nodes_edges_horizontal_right:
                nodes_edges_vertical_below_max_grid_row = max([edge.get_connected_node(node.id).grid_row
                                                               for edge in nodes_edges_horizontal_right])
                nodes_edges_vertical_below_min_grid_row = min([edge.get_connected_node(node.id).grid_row
                                                               for edge in nodes_edges_horizontal_right])
                new_rowspan = nodes_edges_vertical_below_max_grid_row - nodes_edges_vertical_below_min_grid_row + 1

            print(f"colspan node: {node.id}, new_colspan: {new_colspan}, new_rowspan: {new_rowspan}")

            if new_colspan:
                node.grid_colspan = new_colspan
            if new_rowspan:
                node.grid_rowspan = new_rowspan
            # if len(nodes_edges_vertical_below):
            #     node.grid_colspan = len(nodes_edges_vertical_below)
        """

        # Populate table grid with spanned cells
        for node in self.graph.nodes:
            node_start_row = node.grid_row
            node_end_row = node.grid_row + node.grid_rowspan
            node_start_column = node.grid_column
            node_end_column = node.grid_column + node.grid_colspan

            table_grid[node_start_row:node_end_row, node_start_column:node_end_column] = node.id

        print(table_grid)

        """
        # Expand span information to neighboring cells (horizontal)
        num_rows = table_grid.shape[0] - 1
        num_cols = table_grid.shape[1] - 1
        for y in range(num_rows):
            for x in range(num_cols):
                current_value = table_grid[y, x]
                below_cell_value = table_grid[y+1, x]
                right_cell_value = table_grid[y, x+1]
                diagonal_cell_value = table_grid[y+1, x+1]

                if right_cell_value != -1 and right_cell_value != -2:
                    if below_cell_value == -1 and right_cell_value == diagonal_cell_value:
                        table_grid[y+1, x] = current_value

                if below_cell_value != -1 and below_cell_value != -2:
                    if right_cell_value == -1 and below_cell_value == diagonal_cell_value:
                        table_grid[y, x+1] = current_value
        """
        """
        for node in self.graph.nodes:
            node.grid_rowspan = np.amax((table_grid == node.id).sum(axis=0))
            node.grid_colspan = np.amax((table_grid == node.id).sum(axis=1))

        print(table_grid)
        """
        return table_grid

    def fall_vertical_cc(self):
        if not self.vertical_ccs:
            return

        self.vertical_ccs.sort(key=lambda item: item.x)

        # [[(start_row1, end_row1, Row1), (start_row2, end_row2, Row2)],
        #  [(start_row3, end_row3, Row3]]
        [(min_x, min_y), (max_x, max_y)] = self.vertical_ccs[0].bbox
        columns = {0: [(min_y, max_y, self.vertical_ccs[0])]}

        last_column_idx = 0
        for vertical_cc in self.vertical_ccs[1:]:
            [(min_x, min_y), (max_x, max_y)] = vertical_cc.bbox
            tested_column_idx = last_column_idx

            while tested_column_idx >= 0:
                column = columns[tested_column_idx]

                overlap = False
                for (start_row, end_row, Row) in column:
                    cc_vertical_span = (min_y, max_y)
                    if self.ranges_overlap((start_row, end_row), cc_vertical_span):
                        overlap = True

                if overlap:
                    break

                tested_column_idx -= 1

            if (tested_column_idx + 1) not in columns:
                last_column_idx += 1

            columns[tested_column_idx + 1] = columns.get(tested_column_idx + 1, [])
            columns[tested_column_idx + 1] += [(min_y, max_y, vertical_cc)]

        for column_idx in columns:
            for (_, _, vertical_cc) in columns[column_idx]:
                vertical_cc.set_column(column_idx)
        # print(columns)

    def fall_horizontal_cc(self):
        if not self.horizontal_ccs:
            return

        self.horizontal_ccs.sort(key=lambda item: item.y)

        # [[(start_column1, end_column1, Row1), (start_column2, end_column2, Row2)],
        #  [(start_column3, end_column3, Row3]]
        [(min_x, min_y), (max_x, max_y)] = self.horizontal_ccs[0].bbox
        rows = {0: [(min_x, max_x, self.horizontal_ccs[0])]}

        last_row_idx = 0
        for horizontal_cc in self.horizontal_ccs[1:]:
            [(min_x, min_y), (max_x, max_y)] = horizontal_cc.bbox
            tested_column_idx = last_row_idx

            while tested_column_idx >= 0:
                row = rows[tested_column_idx]

                overlap = False
                for (start_column, end_column, Row) in row:
                    cc_horizontal_span = (min_x, max_x)
                    if self.ranges_overlap((start_column, end_column), cc_horizontal_span):
                        overlap = True

                if overlap:
                    break

                tested_column_idx -= 1

            if (tested_column_idx + 1) not in rows:
                last_row_idx += 1

            rows[tested_column_idx + 1] = rows.get(tested_column_idx + 1, [])
            rows[tested_column_idx + 1] += [(min_x, max_x, horizontal_cc)]

        for row_idx in rows:
            for (_, _, horizontal_cc) in rows[row_idx]:
                horizontal_cc.set_row(row_idx)
        print(rows)

    def ranges_overlap(self, a, b):
        # Source: https://stackoverflow.com/questions/2953967/built-in-function-for-computing-overlap-in-python
        overlap_range = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if overlap_range == 0:
            return False
        else:
            return True

    def get_continuous_component(self, current_node, unvisited_edges, discovered_nodes):
        discovered_nodes += [current_node]

        available_edges = [edge for edge in unvisited_edges
                           if edge.node1.id == current_node.id or edge.node2.id == current_node.id]

        if not available_edges:
            return

        for available_edge in available_edges:
            if available_edge in unvisited_edges:
                unvisited_edges.remove(available_edge)
            next_node = available_edge.node1 if available_edge.node1.id != current_node.id else available_edge.node2
            self.get_continuous_component(next_node,
                                          unvisited_edges,
                                          discovered_nodes)

    def lock_edges(self):
        # Create dictionary that maps node id to node's edges
        nodes_edges = {}
        for edge in self.graph.edges:
            nodes_edges[edge.node1.id] = nodes_edges.get(edge.node1.id, [])
            nodes_edges[edge.node1.id] += [edge]

            nodes_edges[edge.node2.id] = nodes_edges.get(edge.node2.id, [])
            nodes_edges[edge.node2.id] += [edge]

        for node in self.graph.nodes:
            # Lock vertical edges
            nodes_vertical_edges = [node for node in nodes_edges[node.id] if node.type == "vertical"]
            nodes_vertical_edges_below = [edge for edge in nodes_vertical_edges
                                          if edge.get_connected_node(node.id).grid_y > node.grid_y]
            nodes_vertical_edges_above = [edge for edge in nodes_vertical_edges
                                          if edge.get_connected_node(node.id).grid_y <= node.grid_y]

            nodes_vertical_edges_below.sort(key=lambda item: item.get_connected_node(node.id).grid_x)
            if len(nodes_vertical_edges_below) > 1:
                for nodes_vertical_edge_below in nodes_vertical_edges_below[1:]:
                    nodes_vertical_edge_below.gridify_locked = True

            nodes_vertical_edges_above.sort(key=lambda item: item.get_connected_node(node.id).grid_x)
            if len(nodes_vertical_edges_above) > 1:
                for nodes_vertical_edge_above in nodes_vertical_edges_above[1:]:
                    nodes_vertical_edge_above.gridify_locked = True

            # Lock horizontal edges
            nodes_horizontal_edges = [node for node in nodes_edges[node.id] if node.type == "horizontal"]
            nodes_horizontal_edges_right = [edge for edge in nodes_horizontal_edges
                                            if edge.get_connected_node(node.id).grid_x > node.grid_x]
            nodes_horizontal_edges_left = [edge for edge in nodes_horizontal_edges
                                           if edge.get_connected_node(node.id).grid_x <= node.grid_x]

            nodes_horizontal_edges_left.sort(key=lambda item: item.get_connected_node(node.id).grid_y)
            if len(nodes_horizontal_edges_left) > 1:
                for nodes_horizontal_edge_left in nodes_horizontal_edges_left[1:]:
                    nodes_horizontal_edge_left.gridify_locked = True

            nodes_horizontal_edges_right.sort(key=lambda item: item.get_connected_node(node.id).grid_y)
            if len(nodes_horizontal_edges_right) > 1:
                for nodes_horizontal_edge_right in nodes_horizontal_edges_right[1:]:
                    nodes_horizontal_edge_right.gridify_locked = True

    def straighten_edges(self, current_node, previous_node, active_edge, unvisited_edges):
        if not active_edge.gridify_locked:
            if active_edge.type == "vertical":
                current_node.grid_x = previous_node.grid_x
            elif active_edge.type == "horizontal":
                current_node.grid_y = previous_node.grid_y

        available_edges = [edge for edge in unvisited_edges
                           if edge.node1.id == current_node.id or edge.node2.id == current_node.id]

        for available_edge in available_edges:
            if available_edge in unvisited_edges:
                unvisited_edges.remove(available_edge)
            next_node = available_edge.node1 if available_edge.node1.id != current_node.id else available_edge.node2
            self.straighten_edges(next_node,
                                  current_node,
                                  available_edge,
                                  unvisited_edges)

    def remove_isolated_nodes(self):
        graph_nodes_ids = []
        for edge in self.graph.edges:
            graph_nodes_ids += [edge.node1.id, edge.node2.id]

        self.graph.nodes = [node for node in self.graph.nodes if node.id in graph_nodes_ids]

    def remove_no_relationship(self):
        self.graph.edges = [edge for edge in self.graph.edges
                            if edge.type != "no-relationship"]

    def remove_symmetrical_edges(self):
        to_keep = []
        for edge in self.graph.edges:
            # Do not keep reflexive edges
            if edge.node1.id == edge.node2.id:
                continue

            if edge.type == "horizontal":
                if edge.node2.bbox["center"][0] >= edge.node1.bbox["center"][0]:
                    if (edge.node2.id, edge.node1.id) not in to_keep:
                        to_keep += [(edge.node1.id, edge.node2.id)]
            elif edge.type == "vertical":
                if edge.node2.bbox["center"][1] >= edge.node1.bbox["center"][1]:
                    if (edge.node2.id, edge.node1.id) not in to_keep:
                        to_keep += [(edge.node1.id, edge.node2.id)]
            else:
                if (edge.node2.id, edge.node1.id) not in to_keep:
                    to_keep += [(edge.node1.id, edge.node2.id)]

        self.graph.edges = [edge for edge in self.graph.edges
                            if (edge.node1.id, edge.node2.id) in to_keep]

    def merge_cell(self):
        cell_edges = [edge for edge in self.graph.edges if edge.type == "cell"]

        nodes_to_remove = []
        for cell_edge in cell_edges:
            # Skip reflexive edges
            if cell_edge.node1.id == cell_edge.node2.id:
                continue

            # -- Join bounding boxes --------------------------------------------------
            node1_bbox = cell_edge.node1.grid_bbox
            node2_bbox = cell_edge.node2.grid_bbox

            max_x = max([x for x, _ in (node1_bbox + node2_bbox)])
            max_y = max([y for _, y in (node1_bbox + node2_bbox)])
            min_x = min([x for x, _ in (node1_bbox + node2_bbox)])
            min_y = min([y for _, y in (node1_bbox + node2_bbox)])

            cell_edge.node1.grid_bbox = [(min_x, min_y), (max_x, max_y)]
            # -------------------------------------------------------------------------

            # -- Join edge connections ------------------------------------------------
            cell_node1 = cell_edge.node1
            cell_node2 = cell_edge.node2
            nodes_to_remove += [cell_node2.id]

            for edge in self.graph.edges:
                if edge.node1.id == cell_node2.id:
                    edge.node1 = cell_node1
                elif edge.node2.id == cell_node2.id:
                    edge.node2 = cell_node1
            # -------------------------------------------------------------------------


        # Remove all cell edges from the graph
        for edge in cell_edges:
            if edge in self.graph.edges:
                self.graph.edges.remove(edge)

        # Remove all nodes that are no longer needed (merged cells)
        self.graph.nodes = [node for node in self.graph.nodes
                            if node.id not in nodes_to_remove]

    def remove_transitive_edges(self):
        horizontal_edges = [edge for edge in self.graph.edges
                            if edge.type == "horizontal"]
        vertical_edges = [edge for edge in self.graph.edges
                          if edge.type == "vertical"]

        horizontal_edges_ids = list(set([(edge.node1.id, edge.node2.id) for edge in horizontal_edges]))
        vertical_edges_ids = list(set([(edge.node1.id, edge.node2.id) for edge in vertical_edges]))

        horizontal_graph = nx.DiGraph(horizontal_edges_ids)
        vertical_graph = nx.DiGraph(vertical_edges_ids)

        # print(list(nx.simple_cycles(vertical_graph)))
        horizontal_graph_list = list(nx.DiGraph(horizontal_edges_ids).edges)
        vertical_graph_list = list(nx.DiGraph(vertical_edges_ids).edges)

        reduced_horizontal_graph = list(nx.transitive_reduction(horizontal_graph).edges)
        reduced_vertical_graph = list(nx.transitive_reduction(vertical_graph).edges)

        horizontal_to_remove = set(horizontal_graph_list).difference(set(reduced_horizontal_graph))
        vertical_to_remove = set(vertical_graph_list).difference(set(reduced_vertical_graph))
        to_remove = horizontal_to_remove.union(vertical_to_remove)

        self.graph.edges = [edge for edge in self.graph.edges
                            if (edge.node1.id, edge.node2.id) not in to_remove]

    def nodes_to_grid(self): # Deprecated
        model = GrapOptimModel(self.graph)
        # TODO: Zkus SGD + Vyzkouset zacit s vysokym LR a pak postupne snizovat
        optimizer = torch.optim.Adam(model.parameters(), lr=1)
        for _ in range(10000):
            optimizer.zero_grad()

            output = model()
            loss = GrapOptimModel.loss(output)
            loss.backward()
            optimizer.step()

        for node in self.graph.nodes:
            node.x = int(torch.floor(model.node_positions_x[str(node.id)]))
            node.y = int(torch.floor(model.node_positions_y[str(node.id)]))


class VerticalCC(object):
    def __init__(self, nodes):
        self.nodes = nodes
        self.x = self.nodes[0].grid_x

        nodes_bboxes = [node.grid_bbox for node in self.nodes]
        min_y = min([min_y for [(_, min_y), (_, _)] in nodes_bboxes])
        max_y = max([max_y for [(_, _), (_, max_y)] in nodes_bboxes])
        min_x = self.x - 1
        max_x = self.x + 1
        self.bbox = [(min_x, min_y), (max_x, max_y)]

    def set_x(self, x):
        diff_x = self.x - x
        self.x = x

        [(min_x, min_y), (max_x, max_y)] = self.bbox
        self.bbox = [(min_x + diff_x, min_y), (max_x + diff_x, max_y)]

    def expand_bbox(self, nodeid_to_edges):
        cc_max_y = None
        cc_min_y = None
        for node in self.nodes:
            nodes_edges = nodeid_to_edges[node.id]
            connected_nodes_horizontal = [edge.get_connected_node(node.id) for edge in nodes_edges
                                        if edge.type == "horizontal"]

            if not connected_nodes_horizontal:
                continue

            max_y = max([node.grid_y for node in connected_nodes_horizontal])
            min_y = min([node.grid_y for node in connected_nodes_horizontal])

            cc_max_y = max_y if cc_max_y is None else cc_max_y
            cc_min_y = min_y if cc_min_y is None else cc_min_y
            cc_max_y = max_y if cc_max_y < max_y else cc_max_y
            cc_min_y = min_y if cc_min_y > min_y else cc_min_y

        if cc_max_y and cc_min_y:
            [(min_x, min_y), (max_x, max_y)] = self.bbox
            self.bbox = [(min_x, cc_min_y), (max_x, cc_max_y)]

    def set_column(self, column):
        for node in self.nodes:
            node.grid_column = column

    def __repr__(self):
        ids = [node.id for node in self.nodes]
        return f"<Column: {ids}>"


class HorizontalCC(object):
    def __init__(self, nodes):
        self.nodes = nodes
        self.y = self.nodes[0].grid_y

        nodes_bboxes = [node.grid_bbox for node in self.nodes]
        min_x = min([min_x for [(min_x, _), (_, _)] in nodes_bboxes])
        max_x = max([max_x for [(_, _), (max_x, _)] in nodes_bboxes])
        min_y = self.y - 1
        max_y = self.y + 1

        self.bbox = [(min_x, min_y), (max_x, max_y)]

    def set_y(self, y):
        diff_y = self.y - y
        self.y = y

        [(min_x, min_y), (max_x, max_y)] = self.bbox
        self.bbox = [(min_x, min_y + diff_y), (max_x, max_y + diff_y)]

    def set_row(self, row):
        for node in self.nodes:
            node.grid_row = row

    def expand_bbox(self, nodeid_to_edges):
        cc_max_x = None
        cc_min_x = None
        for node in self.nodes:
            nodes_edges = nodeid_to_edges[node.id]
            connected_nodes_vertical = [edge.get_connected_node(node.id) for edge in nodes_edges
                                        if edge.type == "vertical"]

            if not connected_nodes_vertical:
                continue

            max_x = max([node.grid_x for node in connected_nodes_vertical])
            min_x = min([node.grid_x for node in connected_nodes_vertical])

            cc_max_x = max_x if cc_max_x is None else cc_max_x
            cc_min_x = min_x if cc_min_x is None else cc_min_x
            cc_max_x = max_x if cc_max_x < max_x else cc_max_x
            cc_min_x = min_x if cc_min_x > min_x else cc_min_x

        if cc_max_x and cc_min_x:
            [(min_x, min_y), (max_x, max_y)] = self.bbox
            self.bbox = [(cc_min_x, min_y), (cc_max_x, max_y)]

    def get_leftmost_node(self):
        self.nodes.sort(key=lambda item: item.x)
        return self.nodes[0]

    def __repr__(self):
        ids = [node.id for node in self.nodes]
        return f"<Row: {ids}>"

class Graph2Text2(object):
    def __init__(self):
        self.nodes_edges = {}

    def populate_nodes_edges(self):
        pass

    def graph_gridification(self, current_node, visited_nodes, previous_nodes_position, traversed_edge_type):
        pass


class GrapOptimModel(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.node_positions_x = torch.nn.ParameterDict({})
        self.node_positions_y = torch.nn.ParameterDict({})
        for node in graph.nodes:
            self.node_positions_x[str(node.id)] = torch.nn.Parameter(torch.tensor(float(node.bbox["center"][0])))
            self.node_positions_y[str(node.id)] = torch.nn.Parameter(torch.tensor(float(node.bbox["center"][1])))

    def forward(self):
        horizontal_error = torch.tensor(0.)
        for edge in self.graph.edges:
            if edge.type != "horizontal":
                continue
            node1_x = self.node_positions_x[str(edge.node1.id)]
            node2_x = self.node_positions_x[str(edge.node2.id)]
            horizontal_error += torch.abs((node1_x - node2_x))  # ** 2

        vertical_error = torch.tensor(0.)
        for edge in self.graph.edges:
            if edge.type != "vertical":
                continue
            node1_y = self.node_positions_y[str(edge.node1.id)]
            node2_y = self.node_positions_y[str(edge.node2.id)]
            vertical_error += torch.abs((node1_y - node2_y))  # ** 2

        """
        horizontal_decimal_error = torch.tensor(0.)
        for node in self.graph.nodes:
            node1_x = self.node_positions_x[str(node.id)]
            horizontal_decimal_error += (torch.floor(node1_x) - node1_x) ** 2
            # horizontal_decimal_error += [(torch.floor(node1_x) - node1_x) ** 2]
        # horizontal_decimal_error = torch.sum(torch.tensor(horizontal_decimal_error))
        """

        return horizontal_error + vertical_error

    @staticmethod
    def loss(output):
        return output ** 2

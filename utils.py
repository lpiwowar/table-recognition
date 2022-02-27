def coords_string_to_tuple_list(coords_string: str) -> [(int, int)]:
    """
    A function that transforms string with coordinates of 2D points
    to list of tuples.

    Example of 2D coordinate string:
        "1,1 2,2 3,3 4,4 ..."

    Example of list of tuples:
         [(1,1), (2,2), (3,3), (4,4), ...]

    :type coords_string:  String
    :param coords_string: String with coordinates of 2D points
    :return:              A list of int tuples
    """

    str_2d_points = coords_string.split(" ")
    return [(int(str_2d_point.split(",")[0]), int(str_2d_point.split(",")[1]))
            for str_2d_point in str_2d_points]


def tuple_list_to_coords_string(tuple_list: [(int, int)]) -> str:
    """
    Inverse function to coords_string_to_tuple_list() that transforms a list
    of integer tuples to string.

    Example of integer tuples:
        [(1,1), (2,2), (3,3), (4,4), ...]

    Example of the output string:
        "1,1 2,2 3,3 4,4 ..."

    :type tuple_list:   List of integer tuples
    :param tuple_list:  List of integer tuples
    :return:            String that contains the transformed list of integer tuples
    """

    return " ".join([str(number_fst) + "," + str(number_nd)
                     for number_fst, number_nd in tuple_list])

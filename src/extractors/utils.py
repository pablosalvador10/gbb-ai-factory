def get_container_and_blob_name_from_url(blob_url: str) -> tuple:
    """
    Retrieves the container name and the blob name from a blob URL.

    The container name is the segment immediately after the host name.
    The blob name is the path after the container name.

    :param blob_url: The blob URL.
    :return: A tuple containing the container name and the blob name.
    """
    from urllib.parse import urlparse

    # Parse the URL to extract the path
    parsed_url = urlparse(blob_url)
    path_segments = parsed_url.path.lstrip('/').split('/')

    # The container name is the first path segment
    container_name = path_segments[0]

    # The blob name is the rest of the path after the container name
    blob_name = '/'.join(path_segments[1:])

    return container_name, blob_name


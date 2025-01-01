import os
GMS_HOME = os.getenv('GMS_HOME')
def get_gms_path(*paths_to_join):
    joined_path = GMS_HOME
    for path in paths_to_join:
        joined_path = os.path.join(joined_path, path.lstrip('/'))
    return joined_path

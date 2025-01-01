import os
GMS_HOME = os.getenv('GMS_HOME')
def get_gms_path(*paths_to_join):
    if paths_to_join[0].startswith(GMS_HOME):        
        joined_path = paths_to_join[0]
    else:
        joined_path = GMS_HOME
    for path in paths_to_join[1:]:
        joined_path = os.path.join(joined_path, path.lstrip('/'))
    return joined_path

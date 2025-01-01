import os
GMS_HOME = os.getenv('GMS_HOME')
def get_gms_path(relative_path):
    return os.path.join(GMS_HOME, relative_path)

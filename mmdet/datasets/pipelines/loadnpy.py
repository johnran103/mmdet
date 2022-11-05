import mmcv
from mmcv import BaseStorageBackend, FileClient

@FileClient.register_backend('npy')
class RemoteUrlBackend(BaseStorageBackend):

    def __init__(self):
        pass

    def get(self, filepath):
        filepath = str(filepath)

        # 本地相对路径替换为远程 url
        for k, v in self.path_mapping.items():
            filepath = filepath.replace(k, v)

        # 拉取远程图片(可能比较复杂)
        value_buf = self._get_remote_image()

        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError

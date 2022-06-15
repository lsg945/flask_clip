class User(object):
    def __init__(self, id:int, username:str, password:str) -> None:
        self.id = id
        self.username = username
        self.password = password
    def __str__(self) -> str:
        return "{}".format(self.id)
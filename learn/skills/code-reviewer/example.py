# 待审查的示例代码

def calculate_average(numbers):
    if len(numbers) == 0:
        return 0
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)


def get_user_data(user_id):
    # TODO: 从数据库获取用户数据
    query = "SELECT * FROM users WHERE id = " + user_id
    return query


class UserManager:
    def __init__(self):
        self.users = []

    def add_user(self, user):
        self.users.append(user)

    def get_user(self, index):
        return self.users[index]

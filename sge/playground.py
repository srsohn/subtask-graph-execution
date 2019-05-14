from .utils import TYPE_PICKUP, TYPE_TRANSFORM, KEY, MOVE_ACTS


class Playground(object):
    def __init__(self):
        # map
        self.env_name = 'playground'
        nb_block = [0, 4]
        nb_water = [0, 0]

        # operation
        operation_list = {
            KEY.PICKUP: dict(name='pickup', oper_type=TYPE_PICKUP, key='p'),
            KEY.TRANSFORM: dict(name='transform', oper_type=TYPE_TRANSFORM, key='t'),
        }

        # object
        obj_list = []
        obj_list.append(dict(imgname='cow.png', name='cow', pickable=True,
                             transformable=True, oid=0, outcome=8,
                             updateable=True, speed=0.1))
        obj_list.append(dict(imgname='duck.png', name='duck', pickable=True,
                             transformable=True, oid=1, outcome=8,
                             updateable=True, speed=0.2))
        obj_list.append(dict(imgname='milk.png', name='milk', pickable=True,
                             transformable=True, oid=2, outcome=8,
                             updateable=False, speed=0))
        obj_list.append(dict(imgname='chest.png', name='chest', pickable=True,
                             transformable=True, oid=3, outcome=8,
                             updateable=False, speed=0))
        obj_list.append(dict(imgname='diamond.png', name='diamond',
                             pickable=True, transformable=True, oid=4,
                             outcome=8, updateable=False, speed=0))
        obj_list.append(dict(imgname='steak.png', name='steak', pickable=True,
                             transformable=True, oid=5, outcome=8,
                             updateable=False, speed=0))
        obj_list.append(dict(imgname='egg.png', name='egg', pickable=True,
                             transformable=True, oid=6, outcome=8,
                             updateable=False, speed=0))
        obj_list.append(dict(imgname='heart.png', name='heart', pickable=True,
                             transformable=True, oid=7, outcome=8,
                             updateable=False, speed=0))
        obj_list.append(dict(imgname='ice.png', name='ice', pickable=False,
                             transformable=False, oid=8, outcome=8,
                             updateable=False, speed=0))

        #item = agent+block+water+objects
        item_name_to_iid = dict()
        item_name_to_iid['agent'] = 0
        item_name_to_iid['block'] = 1
        item_name_to_iid['water'] = 2
        for obj in obj_list:
            item_name_to_iid[obj['name']] = obj['oid'] + 3

        # subtask
        subtask_list = []
        subtask_param_to_id = dict()
        subtask_param_list = []
        for oper, val in operation_list.items():
            for j in range(len(obj_list)):
                obj = obj_list[j]
                if (oper == KEY.PICKUP and obj['pickable']) or \
                        (oper == KEY.TRANSFORM and obj['transformable']):
                    item = dict(param=(oper, j), oper=oper, obj=obj,
                                name=val['name']+' '+obj['name'])

                    subtask_list.append(item)
                    subtask_param_list.append((oper, j))
                    subtask_param_to_id[(oper, j)] = len(subtask_list)-1
        nb_obj_type = len(obj_list)
        nb_operation_type = len(operation_list)

        self.operation_list = operation_list
        self.legal_actions = MOVE_ACTS | {KEY.PICKUP, KEY.TRANSFORM}
        self.nb_operation_type = nb_operation_type

        self.object_param_list = obj_list
        self.nb_obj_type = nb_obj_type
        self.item_name_to_iid = item_name_to_iid
        self.nb_block = nb_block
        self.nb_water = nb_water
        self.subtask_list = subtask_list
        self.subtask_param_list = subtask_param_list
        self.subtask_param_to_id = subtask_param_to_id

        self.nb_subtask_type = len(subtask_list)  # 16
        self.width = 10
        self.height = 10
        self.feat_dim = 3*self.nb_subtask_type+1
        self.ranksep = "0.2"

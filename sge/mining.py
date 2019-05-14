from .utils import TYPE_PICKUP, TYPE_TRANSFORM, KEY, MOVE_ACTS


class Mining(object):
    def __init__(self):
        # map
        self.env_name = 'mining'
        nb_block = [1, 3]
        nb_water = [1, 3]

        # object
        obj_list = []
        obj_list.append(dict(
            name='workspace', pickable=False, transformable=True,
            oid=0, outcome=0, unique=True))
        obj_list.append(dict(
            name='furnace', pickable=False, transformable=True,
            oid=1, outcome=1, unique=True))

        obj_list.append(
            dict(name='tree', pickable=True, transformable=False,
                 oid=2, max=3))
        obj_list.append(
            dict(name='stone', pickable=True, transformable=False,
                 oid=3, max=3))
        obj_list.append(
            dict(name='grass', pickable=True, transformable=False,
                 oid=4, max=2))
        obj_list.append(
            dict(name='pig', pickable=True, transformable=False,
                 oid=5, max=1))

        obj_list.append(
            dict(name='coal', pickable=True, transformable=False,
                 oid=6, max=1))
        obj_list.append(
            dict(name='iron', pickable=True, transformable=False,
                 oid=7, max=1))
        obj_list.append(
            dict(name='silver', pickable=True, transformable=False,
                 oid=8, max=1))
        obj_list.append(
            dict(name='gold', pickable=True, transformable=False,
                 oid=9, max=1))
        obj_list.append(
            dict(name='diamond', pickable=True, transformable=False,
                 oid=10, max=3))
        obj_list.append(dict(
            name='jeweler', pickable=False, transformable=True, oid=11,
            outcome=11, unique=True))
        obj_list.append(dict(
            name='lumbershop', pickable=False, transformable=True, oid=12,
            outcome=12, unique=True))

        for obj in obj_list:
            obj['imgname'] = obj['name']+'.png'

        # operation: pickup (type=0) or transform (type=1)
        operation_list = {
            KEY.PICKUP: dict(name='pickup', oper_type=TYPE_PICKUP, key='p'),
            KEY.USE_1: dict(name='use_1', oper_type=TYPE_TRANSFORM, key='1'),
            KEY.USE_2: dict(name='use_2', oper_type=TYPE_TRANSFORM, key='2'),
            KEY.USE_3: dict(name='use_3', oper_type=TYPE_TRANSFORM, key='3'),
            KEY.USE_4: dict(name='use_4', oper_type=TYPE_TRANSFORM, key='4'),
            KEY.USE_5: dict(name='use_5', oper_type=TYPE_TRANSFORM, key='5'),
        }
        # item = agent+block+water+objects
        item_name_to_iid = dict()
        item_name_to_iid['agent'] = 0
        item_name_to_iid['block'] = 1
        item_name_to_iid['water'] = 2
        for obj in obj_list:
            item_name_to_iid[obj['name']] = obj['oid'] + 3

        # subtask
        subtask_list = []
        subtask_list.append(dict(name='Cut wood',   param=(KEY.PICKUP, 2)))
        subtask_list.append(dict(name="Get stone",  param=(KEY.PICKUP, 3)))
        subtask_list.append(
            dict(name="Get string", param=(KEY.PICKUP, 4)))  # 2
        #
        subtask_list.append(
            dict(name="Make firewood", param=(KEY.USE_1, 12)))  # 3
        subtask_list.append(dict(name="Make stick",    param=(KEY.USE_2, 12)))
        subtask_list.append(dict(name="Make arrow",    param=(KEY.USE_3, 12)))
        subtask_list.append(dict(name="Make bow",      param=(KEY.USE_4, 12)))
        #
        subtask_list.append(
            dict(name="Make stone pickaxe", param=(KEY.USE_1, 0)))  # 7
        subtask_list.append(
            dict(name="Hit pig",            param=(KEY.PICKUP, 5)))
        #
        subtask_list.append(
            dict(name="Get coal", param=(KEY.PICKUP, 6)))  # 9
        subtask_list.append(dict(name="Get iron ore",   param=(KEY.PICKUP, 7)))
        subtask_list.append(dict(name="Get silver ore", param=(KEY.PICKUP, 8)))
        #
        subtask_list.append(dict(name="Light furnace",
                                 param=(KEY.USE_1, 1)))  # 12
        #
        subtask_list.append(
            dict(name="Smelt iron",        param=(KEY.USE_2, 1)))  # 13
        subtask_list.append(
            dict(name="Smelt silver",      param=(KEY.USE_3, 1)))
        subtask_list.append(
            dict(name="Bake pork",         param=(KEY.USE_5, 1)))
        #
        subtask_list.append(dict(name="Make iron pickaxe",
                                 param=(KEY.USE_2, 0)))  # 16
        subtask_list.append(
            dict(name="Make silverware",   param=(KEY.USE_3, 0)))
        #
        subtask_list.append(
            dict(name="Get gold ore",      param=(KEY.PICKUP, 9)))  # 18
        subtask_list.append(
            dict(name="Get diamond ore",   param=(KEY.PICKUP, 10)))
        #
        subtask_list.append(
            dict(name="Smelt gold",      param=(KEY.USE_4, 1)))  # 20
        subtask_list.append(
            dict(name="Craft earrings",  param=(KEY.USE_1, 11)))
        subtask_list.append(
            dict(name="Craft rings",     param=(KEY.USE_2, 11)))
        #
        subtask_list.append(
            dict(name="Make goldware",   param=(KEY.USE_4, 0)))  # 23
        subtask_list.append(dict(name="Make bracelet",   param=(KEY.USE_5, 0)))
        subtask_list.append(
            dict(name="Craft necklace",  param=(KEY.USE_3, 11)))
        #
        subtask_param_to_id = dict()
        subtask_param_list = []
        for i in range(len(subtask_list)):
            subtask = subtask_list[i]
            par = subtask['param']
            subtask_param_list.append(par)
            subtask_param_to_id[par] = i
        nb_obj_type = len(obj_list)
        nb_operation_type = len(operation_list)

        self.operation_list = operation_list
        self.legal_actions = MOVE_ACTS | {
            KEY.PICKUP, KEY.USE_1, KEY.USE_2, KEY.USE_3, KEY.USE_4, KEY.USE_5}

        self.nb_operation_type = nb_operation_type

        self.object_param_list = obj_list
        self.nb_obj_type = nb_obj_type
        self.item_name_to_iid = item_name_to_iid
        self.nb_block = nb_block
        self.nb_water = nb_water
        self.subtask_list = subtask_list
        self.subtask_param_list = subtask_param_list
        self.subtask_param_to_id = subtask_param_to_id

        self.nb_subtask_type = len(subtask_list)
        self.width = 10
        self.height = 10
        self.feat_dim = 3*len(subtask_list)+1
        self.ranksep = "0.1"
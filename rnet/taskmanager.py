from itertools import count
from rnet.env import _QGIS_AVAILABLE, require_qgis

if _QGIS_AVAILABLE:
    from qgis.core import QgsApplication
    from qgis.utils import iface


queue = []
task_counter = count(0)


@require_qgis
def next_task():
    global queue
    try:
        task = queue.pop(0)
    except IndexError:
        iface.statusBarIface().showMessage('Ready')
    else:
        iface.statusBarIface().showMessage(task.description())
        task.taskCompleted.connect(next_task)
        QgsApplication.taskManager().addTask(task)


@require_qgis
def create_and_queue(task, *args, **kwargs):
    '''
    Instantiates `task` and adds it to the task queue.
    '''
    global queue
    task_name = f'task_{next(task_counter)}'
    globals()[task_name] = task(*args, **kwargs)
    queue.append(globals()[task_name])
    if QgsApplication.taskManager().count() == 0:
        next_task()
    return globals()[task_name]

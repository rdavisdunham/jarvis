"""Scheduling constraints, separate from weak organization/routing hints."""
from datetime import timedelta
from .domain import zone


def allowed_intervals(free, windows, timezone):
    if not windows:
        return free
    local = zone(timezone)
    result=[]
    for start,end in free:
        cursor=start
        while cursor<end:
            wall=cursor.astimezone(local)
            clock=wall.strftime("%H:%M")
            allowed=any(
                (wall.weekday() in w["days"] and w["start"]<=clock<w["end"])
                if w["start"]<w["end"] else
                ((wall.weekday() in w["days"] and clock>=w["start"]) or
                 ((wall.weekday()-1)%7 in w["days"] and clock<w["end"]))
                for w in windows)
            next_minute=(cursor+timedelta(minutes=1)).replace(second=0,microsecond=0)
            stop=min(next_minute,end)
            if allowed:
                if result and result[-1][1]==cursor:
                    result[-1]=(result[-1][0],stop)
                else:
                    result.append((cursor,stop))
            cursor=stop
    return result


def task_intervals(request, free, prefs):
    return {t.task_id: (free if request.override_reason else allowed_intervals(
        free,prefs.get("scheduling_windows",{}).get(t.availability,[]),prefs["timezone"]))
        for t in request.tasks}

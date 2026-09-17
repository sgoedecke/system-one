"""Real campaign capture: periodic Qwen plans and asynchronous 35 Hz game."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct
import time
import wave

import numpy as np
from PIL import Image, ImageDraw
import torch
import transformers
import vizdoom as vzd
from system_one import Choice, SystemOne
from demo.labels import LabelSystemOne, label_questions

FPS = 35
MODEL = "Qwen/Qwen3-8B"
STANDING_ORDER = "Reach the campaign exit alive; kill all enemies in the way and collect supplies when needed."
PLAN_HEADS = ("goal", "target")
CONTROL_HEADS = ("dodge", "move", "strafe", "turn", "fire", "weapon", "use")
BUTTONS = [
    vzd.Button.MOVE_FORWARD_BACKWARD_DELTA, vzd.Button.MOVE_BACKWARD,
    vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
    vzd.Button.TURN_LEFT_RIGHT_DELTA, vzd.Button.ATTACK,
    vzd.Button.SELECT_WEAPON2, vzd.Button.SELECT_WEAPON3, vzd.Button.USE,
]
ENDING = " Answer one index digit, no spaces or decimals."
QUESTIONS = {
    "goal": Choice(
        instructions="Choose next strategic goal toward the campaign exit. Prioritize immediate survival, combat, supplies, then exit. Examples: health 30 => choice_index:4; enemy visible and health 100 => choice_index:2; no enemy, health 100, ammo adequate => choice_index:6." + ENDING,
        criteria={
            "Upgrade weapon": "Seek a better weapon when no shotgun is owned and enemies are absent.",
            "Scout": "Explore when the route is obstructed or the current destination was reached without progress.",
            "Kill enemies": "Fight visible enemies when health is above 35 and ammunition remains.",
            "Stock ammo": "Collect ammo when shells are empty and pistol bullets are below 12, or all ammunition is empty.",
            "Restore health": "Find a medkit when health is below 50; urgent at health 35 or less.",
            "Add armor": "Collect nearby armor when health and ammunition are adequate and armor is below 20.",
            "Reach exit": "Advance toward the real level exit when no enemy is visible and health/ammo are adequate.",
        }),
    "target": Choice(instructions="Select destination matching the newly selected planning goal." + ENDING,
                     criteria={"Campaign exit": "Real campaign exit."}),
    "dodge": Choice(
        instructions="Choose evasion while pursuing the ACTIVE goal. Examples: no enemies => choice_index:0; close enemy and left clear => choice_index:1; enemy touching and backward clear => choice_index:3." + ENDING,
        criteria={
            "Carry on": "No immediate enemy within 180 units. Do not strafe merely because an enemy exists.",
            "Dodge left": "Enemy within 180 units and left clearance greater than 70.",
            "Dodge right": "Enemy within 180 units, left blocked, and right clearance greater than 70.",
            "Dodge back": "Enemy within 70 units and backward clearance greater than 70.",
        }),
    "move": Choice(
        instructions="Move toward the ACTIVE target using route bearing, not a newly selected target. Examples: route centered and clear => choice_index:0; route left 80 degrees => choice_index:1; forward blocked and stuck => choice_index:2. Keep advancing through corridors when aligned." + ENDING,
        criteria={
            "Forward": "Route waypoint is within 35 degrees ahead and forward clearance exceeds 45. Approach targets, including distant enemies.",
            "Hold": "Route waypoint is more than 35 degrees away from crosshair; turn before moving. Also hold if destination reached.",
            "Backward": "Stuck against an obstacle and backward clearance exceeds 65.",
        }),
    "strafe": Choice(
        instructions="Default to Hold: normal navigation uses movement and turning, not strafing. Sidestep only to recover when stuck, then return to Hold as soon as movement resumes. One index digit.",
        criteria={
            "Hold": "Status NORMAL, regardless of side clearance. Also hold if both sides BLOCKED.",
            "Strafe left": "Status JAMMED and left CLEAR.",
            "Strafe right": "Status JAMMED and right CLEAR.",
        }),
    "turn": Choice(
        instructions="Turn toward the explicitly described AIM direction. Positive bearing means left. Examples: AIM left 40 degrees => choice_index:0; AIM left 5 => choice_index:1; AIM centered => choice_index:2; AIM right 5 => choice_index:3; AIM right 40 => choice_index:4." + ENDING,
        criteria={
            "Hard left": "AIM is left more than 12 degrees.",
            "Left": "AIM is left between 2 and 12 degrees.",
            "Hold": "AIM is centered within 2 degrees.",
            "Right": "AIM is right between 2 and 12 degrees.",
            "Hard right": "AIM is right more than 12 degrees.",
        }),
    "fire": Choice(
        instructions="Conserve scarce ammunition. Fire ONLY at an actually visible enemy within 7 degrees of crosshair with ammo. The navigation target is NOT an enemy. Examples: no visible enemy => choice_index:1; enemy 30 degrees left => choice_index:1; enemy centered and ammo 2 => choice_index:0." + ENDING,
        criteria={
            "Fire": "A visible living enemy is within 7 degrees of the crosshair and equipped weapon ammunition is positive.",
            "Hold fire": "No visible living enemy is within 7 degrees, or the equipped weapon is empty. Never shoot navigation destinations.",
        }),
    "weapon": Choice(
        instructions="Select weapon under scarcity. Examples: shotgun equipped shells 2 => choice_index:0; shotgun equipped shells 0 bullets 25 => choice_index:1; pistol equipped shells 0 => choice_index:0; pistol equipped shells 4 => choice_index:2." + ENDING,
        criteria={
            "Keep": "Keep shotgun if shells remain; keep pistol if no shells remain.",
            "Pistol": "Switch away from empty shotgun when bullets remain.",
            "Shotgun": "Switch from pistol to owned shotgun if shells are available.",
        }),
    "use": Choice(
        instructions="Interact with campaign doors and exit switches. Examples: door or switch immediately ahead within 90 units => choice_index:0; ordinary corridor => choice_index:1." + ENDING,
        criteria={
            "Use": "A door or exit switch is within 90 units ahead, or stuck at the route doorway while facing it.",
            "Wait": "No door or exit switch immediately ahead.",
        }),
}
ENEMIES = {"DoomImp", "ZombieMan", "Zombieman", "ShotgunGuy", "Demon", "Spectre", "ChaingunGuy",
           "Cacodemon", "HellKnight", "BaronOfHell", "LostSoul", "Revenant", "Arachnotron",
           "Fatso", "PainElemental", "Archvile", "WolfensteinSS"}
QUESTIONS["goal"]=Choice(instructions="Choose the immediate priority from the CURRENT situation, not the old goal. One index digit.",
    criteria={
        "Kill enemies":"Enemy NEAR, health OK or LOW, ammo READY.",
        "Restore health":"Health LOW with no nearby enemy, or health CRITICAL.",
        "Stock ammo":"Ammo EMPTY or SCARCE. Replenish before fighting.",
        "Add armor":"Armor NEEDED nearby, health OK, no nearby enemy, ammo READY.",
        "Reach exit":"Enemies absent or DISTANT; health OK; ammo READY; armor not needed nearby.",
        "Scout":"The destination is reached or route is blocked and player is stuck.",
        "Upgrade weapon":"No shotgun owned, no immediate combat.",
    })
QUESTIONS["move"]=Choice(instructions="Advance toward the ACTIVE destination or stop to rotate? One index digit.",
    criteria={"Forward":"Route aligned ahead AND forward corridor clear.",
              "Hold":"Route NOT aligned ahead, or forward blocked.",
              "Backward":"Stuck at obstacle with clear space behind."})
QUESTIONS["fire"]=Choice(instructions="Is a living enemy in the crosshair, with ammunition to shoot? One index digit.",
    criteria={"Fire":"Crosshair enemy: centered. Equipped ammo positive.",
              "Hold fire":"Crosshair enemy: none or off-target. Or equipped weapon empty."})
QUESTIONS["turn"]=Choice(instructions="Match the AIM direction, not the enemy direction or destination name. One index digit.",
    criteria={"Hard left":"AIM far left.","Left":"AIM moderately left.","Fine left":"AIM slightly left.","Hold":"AIM centered.",
              "Fine right":"AIM slightly right.","Right":"AIM moderately right.","Hard right":"AIM far right."})
ITEMS = {
    2011: ("Health", "Stimpack"), 2012: ("Health", "Medkit"),
    2018: ("Armor", "Armor"), 2019: ("Armor", "Mega armor"),
    2007: ("Ammo", "Bullets"), 2048: ("Ammo", "Bullet box"),
    2008: ("Ammo", "Shells"), 2049: ("Ammo", "Shell box"),
    2001: ("Weapon", "Shotgun"), 2002: ("Weapon", "Chaingun"),
    5: ("Key", "Blue key"), 6: ("Key", "Yellow key"), 13: ("Key", "Red key"),
}


def bearing(x, y, angle, tx, ty):
    return (math.degrees(math.atan2(ty-y, tx-x))-angle+180) % 360-180


def direction(b):
    return "centered" if abs(b) <= 2 else ("left" if b > 0 else "right")


class CampaignMap:
    """Static WAD geometry is navigation observation only, never a control policy."""
    def __init__(self, wad, level, skill=2):
        data = wad.read_bytes()
        _, count, offset = struct.unpack_from("<4sii", data)
        lumps = []
        for i in range(count):
            pos, size, name = struct.unpack_from("<ii8s", data, offset+16*i)
            lumps.append((name.rstrip(b"\0").decode(), data[pos:pos+size]))
        start = next(i for i, (name, _) in enumerate(lumps) if name == level)
        parts = dict(lumps[start+1:start+11])
        vertices = list(struct.iter_unpack("<hh", parts["VERTEXES"]))
        sides = list(struct.iter_unpack("<hh8s8s8sh", parts["SIDEDEFS"]))
        self.sectors = list(struct.iter_unpack("<hh8s8shhh", parts["SECTORS"]))
        self.lines = []
        self.items = []
        for i, (v1, v2, flags, special, tag, front, back) in enumerate(struct.iter_unpack("<HHHHHHH", parts["LINEDEFS"])):
            a, b = vertices[v1], vertices[v2]
            fs = sides[front][-1] if front != 65535 else -1
            bs = sides[back][-1] if back != 65535 else -1
            self.lines.append((*a, *b, flags, special, fs, bs))
            if special in {11, 51, 52, 124}:
                dx, dy = b[0]-a[0], b[1]-a[1]
                size = math.hypot(dx, dy)
                self.items.append({"id": f"exit-{i}", "name": "Campaign exit", "kind": "Exit",
                                   "x": (a[0]+b[0])/2+dy/size*48, "y": (a[1]+b[1])/2-dx/size*48})
        for i, (x, y, angle, kind, flags) in enumerate(struct.iter_unpack("<hhhhh", parts["THINGS"])):
            difficulty_flag=1 if skill<=2 else 2 if skill==3 else 4
            if kind in ITEMS and flags & difficulty_flag and not flags & 16:
                category, name = ITEMS[kind]
                self.items.append({"id": f"item-{i}", "name": f"{name} {i}", "kind": category, "x": x, "y": y})
        self.step = 32
        self.x0 = math.floor(min(v[0] for v in vertices)/32)*32-64
        self.y0 = math.floor(min(v[1] for v in vertices)/32)*32-64
        width = math.ceil((max(v[0] for v in vertices)-self.x0)/32)+3
        height = math.ceil((max(v[1] for v in vertices)-self.y0)/32)+3
        self.grid = np.full((height, width), -1, dtype=np.int32)
        for iy in range(height):
            y = self.y0+iy*32
            hits = []
            for x1,y1,x2,y2,flags,special,fs,bs in self.lines:
                if min(y1,y2) <= y < max(y1,y2):
                    hits.append((x1+(y-y1)*(x2-x1)/(y2-y1), bs if y2>y1 else fs))
            hits.sort()
            for ix in range(width):
                x = self.x0+ix*32
                right = next((sector for hx,sector in hits if hx>x), -1)
                self.grid[iy,ix] = right
        blockers = Image.new("L", (width*4,height*4))
        draw = ImageDraw.Draw(blockers)
        for x1,y1,x2,y2,flags,special,fs,bs in self.lines:
            blocked = bs < 0 or fs < 0 or bool(flags & 1)
            if not blocked:
                a, b = self.sectors[fs], self.sectors[bs]
                blocked = abs(a[0]-b[0]) > 24
            if blocked:
                draw.line(((x1-self.x0)/8,(y1-self.y0)/8,(x2-self.x0)/8,(y2-self.y0)/8), fill=255,width=5)
        mask = np.array(blockers)[::4,::4] > 0
        self.grid[mask] = -1
        self.valid = np.argwhere(self.grid >= 0)
        self.visited = set()
        self.collected = set()
        self.initial_items = None
        self.unavailable = set()
        self.last_path = []

    def cell(self, x, y):
        cell = (round((y-self.y0)/32), round((x-self.x0)/32))
        iy,ix = cell
        if 0<=iy<self.grid.shape[0] and 0<=ix<self.grid.shape[1] and self.grid[cell]>=0:
            return cell
        return tuple(self.valid[np.argmin((self.valid[:,0]-iy)**2+(self.valid[:,1]-ix)**2)])

    def xy(self, cell):
        return self.x0+int(cell[1])*32,self.y0+int(cell[0])*32

    def path(self, x,y,tx,ty):
        start, goal = self.cell(x,y), self.cell(tx,ty)
        queue=[(0,start)]
        costs={start:0}
        parents={}
        closest=start
        closest_distance=math.hypot(start[0]-goal[0],start[1]-goal[1])
        while queue:
            _,node=heapq.heappop(queue)
            distance=math.hypot(node[0]-goal[0],node[1]-goal[1])
            if distance<closest_distance:
                closest,closest_distance=node,distance
            if node==goal:
                result=[node]
                while node in parents:
                    node=parents[node]
                    result.append(node)
                return [self.xy(p) for p in reversed(result)]
            for dy,dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
                nxt=(node[0]+dy,node[1]+dx)
                if not (0<=nxt[0]<self.grid.shape[0] and 0<=nxt[1]<self.grid.shape[1]) or self.grid[nxt]<0:
                    continue
                if dx and dy and (self.grid[node[0],nxt[1]]<0 or self.grid[nxt[0],node[1]]<0):
                    continue
                cost=costs[node]+math.hypot(dx,dy)
                if cost<costs.get(nxt,1e9):
                    costs[nxt]=cost
                    parents[nxt]=node
                    h=math.hypot(nxt[0]-goal[0],nxt[1]-goal[1])
                    heapq.heappush(queue,(cost+h,nxt))
        result=[closest]
        while closest in parents:
            closest=parents[closest]
            result.append(closest)
        return [self.xy(p) for p in reversed(result)]

    def route(self, x,y,target):
        path=self.path(x,y,target["x"],target["y"])
        self.last_path=path
        if not path:
            return (target["x"],target["y"]),None
        index=min(4,len(path)-1)
        return path[index],round((len(path)-1)*32)

    def nearby_interaction(self,x,y,angle):
        candidates=[]
        for x1,y1,x2,y2,flags,special,fs,bs in self.lines:
            if not special:
                continue
            dx,dy=x2-x1,y2-y1
            t=max(0,min(1,((x-x1)*dx+(y-y1)*dy)/(dx*dx+dy*dy)))
            px,py=x1+t*dx,y1+t*dy
            dist=math.hypot(px-x,py-y)
            if dist<90 and abs(bearing(x,y,angle,px,py))<45:
                candidates.append({"special":special,"distance":round(dist)})
        return candidates


def clearance(x,y,angle,sectors):
    dx,dy=math.cos(angle),math.sin(angle)
    distance=4096.
    for sector in sectors:
        for line in sector.lines:
            if not line.is_blocking:
                continue
            sx,sy=line.x2-line.x1,line.y2-line.y1
            det=dx*sy-dy*sx
            if abs(det)<1e-8:
                continue
            qx,qy=line.x1-x,line.y1-y
            t=(qx*sy-qy*sx)/det
            u=(qx*dy-qy*dx)/det
            if t>=0 and 0<=u<=1:
                distance=min(distance,t)
    return round(distance)


def create_game(seed,seconds,level,skill=2,config_path=None):
    game=vzd.DoomGame()
    if config_path is not None:
        game.set_doom_config_path(str(config_path))
    game.set_doom_game_path(str(Path(vzd.__file__).parent/"freedoom2.wad"))
    game.set_doom_map(level)
    game.set_doom_skill(skill)
    game.set_available_buttons(BUTTONS)
    game.set_button_max_value(vzd.Button.TURN_LEFT_RIGHT_DELTA,2)
    game.set_button_max_value(vzd.Button.MOVE_FORWARD_BACKWARD_DELTA,14)
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(True)
    game.set_render_crosshair(True)
    game.set_labels_buffer_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sound_enabled(False)
    game.set_audio_buffer_enabled(True)
    game.set_audio_sampling_rate(vzd.SamplingRate.SR_44100)
    game.set_audio_buffer_size(1)
    game.set_episode_timeout(round((seconds+30)*FPS))
    game.set_seed(seed)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()
    return game


def inventory(game):
    def get(name):
        return int(game.get_game_variable(getattr(vzd.GameVariable,name)))
    return {"health":get("HEALTH"),"armor":get("ARMOR"),"shells":get("AMMO3"),"bullets":get("AMMO2"),
            "selected_weapon":{2:"pistol",3:"shotgun"}.get(get("SELECTED_WEAPON"),"other"),
            "selected_weapon_ammo":get("SELECTED_WEAPON_AMMO"),"own_shotgun":bool(get("WEAPON3"))}


def start_episode(game,skill=2):
    game.new_episode()
    commands=["give shotgun","take Shell 999",f"give Shell {1 if skill==1 else 2}","take Clip 999",f"give Clip {15 if skill==1 else 30}"]
    for command in commands:
        game.send_game_command(command)
    for _ in range(20):
        action=[0]*len(BUTTONS)
        action[7]=1
        game.make_action(action,1)
    inv=inventory(game)
    if inv["shells"]!=2 or inv["bullets"]!=30 or inv["selected_weapon"]!="shotgun":
        raise RuntimeError(f"Starting inventory is not the requested exact loadout: {inv}")
    return inv


def targets_for_goal(goal, enemies, candidates, scout):
    categories={"Upgrade weapon":"Weapon","Stock ammo":"Ammo","Restore health":"Health","Add armor":"Armor","Reach exit":"Exit","Kill enemies":"Enemy"}
    desired=categories.get(goal)
    if desired=="Enemy":
        relevant=[]
        for enemy in sorted(enemies,key=lambda e:e["distance"])[:3]:
            relevant.append(dict(enemy,mode="engage"))
            relevant.append(dict(enemy,id=enemy["id"]+"-approach",name="Approach "+enemy["name"],mode="navigate"))
    else:
        relevant=[p for p in candidates if p["kind"]==desired][:4]
    exits=[p for p in candidates if p["kind"]=="Exit"][:1]
    pool=relevant if relevant else (scout[:3] if goal=="Scout" else exits+scout[:2])
    return [dict(p) for p in {p["id"]:p for p in pool}.values()][:9]


def observe(game,state,nav,goal,target,memory):
    get=lambda name:float(game.get_game_variable(getattr(vzd.GameVariable,name)))
    x,y,angle=get("POSITION_X"),get("POSITION_Y"),get("ANGLE")
    obs=inventory(game)
    obs.update(x=x,y=y,angle=angle,kills=int(get("KILLCOUNT")))
    enemies=[]
    for label in state.labels:
        if label.object_name not in ENEMIES:
            continue
        b=bearing(x,y,angle,label.object_position_x,label.object_position_y)
        enemies.append({"id":f"enemy-{label.object_id}","name":f"{label.object_name} {label.object_id}",
                        "kind":"Enemy","x":label.object_position_x,"y":label.object_position_y,
                        "distance":round(math.hypot(label.object_position_x-x,label.object_position_y-y)),
                        "bearing":round(b,1),"direction":direction(b)})
    enemies.sort(key=lambda e:abs(e["bearing"]))
    obs["enemies"]=enemies
    obs["visible_label_names"]=[label.object_name for label in state.labels]
    nav.visited.add(nav.cell(x,y))
    item_actors={"Stimpack","Medikit","GreenArmor","BlueArmor","Clip","ClipBox","Shell","ShellBox",
                 "Shotgun","Chaingun","BlueCard","YellowCard","RedCard"}
    live_items={(round(actor.position_x),round(actor.position_y)) for actor in state.objects if actor.name in item_actors}
    present={item["id"] for item in nav.items if item["kind"]=="Exit" or (round(item["x"]),round(item["y"])) in live_items}
    if nav.initial_items is None:
        nav.initial_items=present
        nav.unavailable={item["id"] for item in nav.items}-present
    nav.collected=nav.initial_items-present
    obs["unavailable_at_episode_start"]=sorted(nav.unavailable)
    obs["live_item_ids"]=sorted(present)
    candidates=[dict(item) for item in nav.items if item["id"] in present]
    candidates.sort(key=lambda item:math.hypot(item["x"]-x,item["y"]-y))
    obs["nearest_armor_distance"]=min((round(math.hypot(p["x"]-x,p["y"]-y)) for p in candidates if p["kind"]=="Armor"),default=None)
    scout=[]
    for cell in nav.valid[::max(1,len(nav.valid)//100)]:
        c=tuple(cell)
        if c not in nav.visited:
            sx,sy=nav.xy(c)
            distance=math.hypot(sx-x,sy-y)
            if 160<distance<800:
                scout.append({"id":f"scout-{sx}-{sy}","name":f"Corridor {len(scout)+1}",
                              "kind":"Scout","x":sx,"y":sy})
    scout.sort(key=lambda p:math.hypot(p["x"]-x,p["y"]-y))
    obs["target_pools"]={name:targets_for_goal(name,enemies,candidates,scout)
                         for name in QUESTIONS["goal"].criteria}
    for pool in obs["target_pools"].values():
        for candidate in pool:
            candidate["distance"]=round(math.hypot(candidate["x"]-x,candidate["y"]-y))
    obs["targets"]=obs["target_pools"][goal]
    if not obs["targets"]:
        raise RuntimeError("No campaign targets")
    base_id=target["id"].removesuffix("-approach")
    live_enemy=next((e for e in enemies if e["id"]==base_id),None)
    live_target=dict(target,x=live_enemy["x"],y=live_enemy["y"]) if live_enemy else target
    waypoint,length=nav.route(x,y,live_target)
    rb=bearing(x,y,angle,*waypoint)
    target_distance=math.hypot(live_target["x"]-x,live_target["y"]-y)
    chosen_enemy=next((e for e in enemies if e["id"]==target["id"] and target.get("mode")!="navigate"),None)
    aim=chosen_enemy["bearing"] if chosen_enemy and goal=="Kill enemies" else rb
    obs.update(active_goal=goal,active_target=live_target,route_bearing=round(rb,1),
               route_direction=direction(rb),route_distance=length,waypoint=waypoint,
               route_reaches_target=bool(nav.last_path and math.hypot(nav.last_path[-1][0]-live_target["x"],nav.last_path[-1][1]-live_target["y"])<64),
               target_distance=round(target_distance),aim_bearing=round(aim,1),aim_direction=direction(aim),
               visited_cells=len(nav.visited),collected_or_passed=sorted(nav.collected))
    obs["clearance"]={name:clearance(x,y,math.radians(angle+offset),state.sectors)
                      for name,offset in [("forward",0),("left",90),("backward",180),("right",-90)]}
    obs["interactions"]=nav.nearby_interaction(x,y,angle)
    positions=memory["positions"]
    positions.append((x,y))
    if len(positions)>10:
        positions.pop(0)
    obs["stuck"]=len(positions)>=10 and math.hypot(x-positions[0][0],y-positions[0][1])<16
    return obs


def describe(obs):
    enemy=obs["enemies"][0] if obs["enemies"] else None
    combat=(f"Enemy visible: yes, {enemy['name']}, {abs(enemy['bearing'])} degrees {enemy['direction']}, distance {enemy['distance']}. "
            f"Crosshair enemy: {'centered' if abs(enemy['bearing'])<=7 else 'off-target'}."
            if enemy else "Enemy visible: NO. Crosshair enemy: none.")
    aim="centered" if abs(obs["aim_bearing"])<=2 else ("slightly " if abs(obs["aim_bearing"])<=12 else "moderately " if abs(obs["aim_bearing"])<=45 else "far ")+obs["aim_direction"]
    armor=obs["nearest_armor_distance"]
    return (
        f"STANDING ORDER: {STANDING_ORDER}\n"
        f"CURRENT SITUATION: Health {obs['health']}; armor {obs['armor']}. {combat}\n"
        f"Equipped {obs['selected_weapon']}; equipped ammo {obs['selected_weapon_ammo']}; shells {obs['shells']}; pistol bullets {obs['bullets']}; own shotgun {obs['own_shotgun']}. Nearby armor distance: {armor}.\n"
        f"ACTIVE COMMITTED PLAN: {obs['active_goal']} / {obs['active_target']['name']}. All controls follow this plan.\n"
        f"NAVIGATION: Route {'aligned ahead' if abs(obs['route_bearing'])<=35 else 'NOT aligned ahead'}; final destination distance {obs['target_distance']}. "
        f"AIM {aim}, offset {abs(obs['aim_bearing'])} degrees.\n"
        f"Forward corridor {'clear' if obs['clearance']['forward']>45 else 'blocked'}. Wall clearance: "+
        ", ".join(f"{k} {v}" for k,v in obs["clearance"].items())+
        f". Stuck: {obs['stuck']}. Door/switch immediately ahead: {bool(obs['interactions'])}."
    )


def questions_for(obs, label_map=None):
    questions=dict(QUESTIONS)
    enemy=obs["enemies"][0] if obs["enemies"] else None
    closest=min(obs["enemies"],key=lambda e:e["distance"]) if obs["enemies"] else None
    crosshair=("centered" if abs(enemy["bearing"])<=7 else "off-target") if enemy else "none"
    questions["fire"]=Choice(
        instructions=f"TRIGGER observation: living enemy in crosshair = {crosshair}; enemy range {'within 450' if enemy and enemy['distance']<=450 else 'beyond 450 or absent'}; equipped ammunition = {obs['selected_weapon_ammo']}. Ignore navigation AIM. Choose whether to fire at a living enemy. One index digit.",
        criteria={"Fire":"Living enemy centered in crosshair, within 450 units, AND equipped ammunition greater than zero.",
                  "Hold fire":"Enemy off-target, absent, beyond 450 units, or equipped ammunition zero."})
    questions["goal"]=Choice(
        instructions=f"Choose immediate priority, not the old plan. Health {'CRITICAL' if obs['health']<35 else 'LOW' if obs['health']<50 else 'OK'} ({obs['health']}). Enemy {'NEAR' if closest and closest['distance']<=450 else 'DISTANT' if closest else 'ABSENT'}. Ammo {'EMPTY' if obs['shells']==0 and obs['bullets']==0 else 'SCARCE' if obs['shells']==0 and obs['bullets']<12 else 'READY'}. Armor {'NEEDED nearby' if obs['armor']<20 and obs['nearest_armor_distance'] is not None and obs['nearest_armor_distance']<400 else 'not needed nearby'} ({obs['armor']}). One index digit.",
        criteria=QUESTIONS["goal"].criteria)
    questions["dodge"]=Choice(
        instructions=f"Evasion facts: threat {'DANGER' if closest and closest['distance']<180 else 'SAFE'}. Left {'CLEAR' if obs['clearance']['left']>70 else 'BLOCKED'}, right {'CLEAR' if obs['clearance']['right']>70 else 'BLOCKED'}, backward {'CLEAR' if obs['clearance']['backward']>70 else 'BLOCKED'}. One index digit.",
        criteria={"Carry on":"Threat SAFE.","Dodge left":"Threat DANGER and left CLEAR.",
                  "Dodge right":"Threat DANGER, left BLOCKED, right CLEAR.",
                  "Dodge back":"Threat DANGER, sideways BLOCKED, backward CLEAR."})
    questions["strafe"]=Choice(
        instructions=f"RECOVERY STATUS: {'JAMMED' if obs['stuck'] else 'NORMAL'}. Left {'CLEAR' if obs['clearance']['left']>70 else 'BLOCKED'}; right {'CLEAR' if obs['clearance']['right']>70 else 'BLOCKED'}. NORMAL means the player is not stuck: choose Hold regardless of side clearance. JAMMED means the player is stuck: choose a clear side, preferring left if both sides are clear. Never strafe into a blocked side. Return to Hold when status becomes NORMAL. Combat evasion is handled separately by dodge. One index digit.",
        criteria=QUESTIONS["strafe"].criteria)
    questions["weapon"]=Choice(
        instructions=f"Which weapon should be equipped? Shotgun {'LOADED' if obs['shells']>0 else 'EMPTY'}, shells {obs['shells']}; pistol bullets {obs['bullets']}. One index digit.",
        criteria={"Pistol":"Shotgun EMPTY.","Shotgun":"Shotgun LOADED."})
    aim="centered" if abs(obs["aim_bearing"])<=2 else ("slightly " if abs(obs["aim_bearing"])<=12 else "moderately " if abs(obs["aim_bearing"])<=45 else "far ")+obs["aim_direction"]
    questions["turn"]=Choice(
        instructions=f"The committed plan's AIM is {aim}, offset {abs(obs['aim_bearing'])} degrees. Match this AIM direction. One index digit.",
        criteria=QUESTIONS["turn"].criteria)
    questions["move"]=Choice(
        instructions=f"Route toward committed destination is {'aligned ahead' if abs(obs['route_bearing'])<=35 else 'NOT aligned ahead'}. Forward is {'clear' if obs['clearance']['forward']>45 else 'blocked'}. Stuck: {obs['stuck']}. Choose movement. One index digit.",
        criteria=QUESTIONS["move"].criteria)
    questions["target"]=Choice(
        instructions=f"Selected planning goal: {obs['active_goal']}. Select the target. For combat, choose Approach for a distant enemy beyond 220 units; engage an enemy within 220. For other goals choose its matching item or exit." + ENDING,
        criteria={p["name"]:(f"Navigate toward this {'DISTANT' if p['distance']>220 else 'NEAR'} enemy, distance {p['distance']}; approach before aiming." if p.get("mode")=="navigate" else
                             f"Stand and aim at this {'NEAR' if p['distance']<=220 else 'DISTANT'} enemy, distance {p['distance']}." if p.get("mode")=="engage" else
                             f"{p['kind']} destination, distance {p['distance']} units.") for p in obs["targets"]})
    return label_questions(questions, label_map) if label_map is not None else questions


def controls(answers):
    a={k:v["choice"] for k,v in answers.items()}
    sideways=a["strafe"] if a["dodge"]=="Carry on" else a["dodge"]
    return [14*int(a["move"]=="Forward"),int(a["move"]=="Backward" or a["dodge"]=="Dodge back"),
            int(sideways in {"Strafe left","Dodge left"}),int(sideways in {"Strafe right","Dodge right"}),
            {"Hard left":-2.0,"Left":-.7,"Fine left":-.15,"Hold":0,"Fine right":.15,"Right":.7,"Hard right":2.0}[a["turn"]],
            int(a["fire"]=="Fire"),int(a["weapon"]=="Pistol"),int(a["weapon"]=="Shotgun"),int(a["use"]=="Use")]


def infer(engine,obs,cache,heads,clock=time.perf_counter):
    text=describe(obs)
    if tuple(heads) == ("target",):
        # The snapshot's route still describes the previous destination.
        text="\n".join(text.splitlines()[:3])
        text+=f"\nNEW SELECTED PLANNING GOAL: {obs['active_goal']}."
    questions={key:q for key,q in questions_for(obs, getattr(engine, "label_map", None)).items()
               if key in heads}
    start=clock()
    result=engine.system_one(text,questions,cache_prefix=cache)
    latency=(clock()-start)*1000
    if set(result.answers) != set(heads):
        raise ValueError(f"Expected answers for {heads}, got {tuple(result.answers)}")
    answers={key:{"choice":answer.choice,"probabilities":answer.probabilities} for key,answer in result.answers.items()}
    for key,answer in answers.items():
        if answer["choice"] not in questions[key].criteria:
            raise ValueError(f"Invalid {key} choice: {answer['choice']}")
    return {"answers":answers,"latency_ms":latency,"model_state":text,
            "questions":{k:{"instructions":q.instructions,"criteria":q.criteria} for k,q in questions.items()}}


def infer_request(engine,request,cache,clock=time.perf_counter):
    """One worker owns both planning stages; its snapshots contain no game objects."""
    started=clock()
    obs=request["observation"]
    kind=request["kind"]
    if kind=="plan":
        goal_result=infer(engine,obs,cache,("goal",),clock)
        goal=goal_result["answers"]["goal"]["choice"]
        target_obs=dict(obs,active_goal=goal,targets=obs["target_pools"][goal])
        if not target_obs["targets"]:
            raise ValueError(f"No candidates for selected goal {goal}")
        target_result=infer(engine,target_obs,cache,("target",),clock)
        target_name=target_result["answers"]["target"]["choice"]
        targets=[p for p in target_obs["targets"] if p["name"]==target_name]
        if len(targets)!=1:
            raise ValueError(f"Target must identify one actual candidate: {target_name}")
        result={"answers":{**goal_result["answers"],**target_result["answers"]},
                "questions":{**goal_result["questions"],**target_result["questions"]},
                "model_state":{"goal":goal_result["model_state"],"target":target_result["model_state"]},
                "goal_latency_ms":goal_result["latency_ms"],"target_latency_ms":target_result["latency_ms"],
                "control_latency_ms":None,"active_goal":goal,"active_target":targets[0]}
        result["latency_ms"]=result["goal_latency_ms"]+result["target_latency_ms"]
    elif kind=="control":
        result=infer(engine,obs,cache,CONTROL_HEADS,clock)
        result.update(control_latency_ms=result["latency_ms"],goal_latency_ms=None,target_latency_ms=None)
    else:
        raise ValueError(f"Unknown inference kind {kind}")
    completed=clock()
    result.update(kind=kind,episode=request["episode"],plan_id=request["plan_id"],
                  evaluated_heads=list(PLAN_HEADS if kind=="plan" else CONTROL_HEADS),
                  observation_frame=request["frame"],worker_started=started,worker_completed=completed,
                  worker_wall_ms=(completed-started)*1000,
                  planning_latency_ms=(completed-started)*1000 if kind=="plan" else None)
    return result


def positive_integer(value):
    try:
        number=int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if number<=0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


class PlanCadence:
    """Advance only when a completed control inference is actually applied."""
    def __init__(self,every):
        self.every=positive_integer(every)
        self.plan_id=0
        self.reset(1)

    def reset(self,episode):
        self.episode=episode
        self.has_plan=False
        self.controls_since_plan=0
        self.answers={}
        self.questions={}
        self.last_control_applied=None
        self.last_control_frame=None

    @property
    def next_kind(self):
        return "plan" if not self.has_plan or self.controls_since_plan>=self.every else "control"

    def accepts(self,result):
        return result["episode"]==self.episode and result["plan_id"]==self.plan_id

    def commit(self,result,applied_at,frame):
        if not self.accepts(result):
            return False
        if result["kind"]!=self.next_kind:
            raise ValueError("Inference result does not match the scheduled cadence")
        if result["kind"]=="plan":
            self.plan_id+=1
            self.has_plan=True
            self.controls_since_plan=0
        else:
            self.controls_since_plan+=1
            self.last_control_applied=applied_at
            self.last_control_frame=frame
        self.answers.update(result["answers"])
        self.questions.update(result["questions"])
        return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seconds",type=float,default=100)
    parser.add_argument("--plan-every",type=positive_integer,default=3,
                        help="Replan after this many applied, completed control inferences (default: 3)")
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--level",default="MAP01")
    parser.add_argument("--cache-prefix",action="store_true")
    parser.add_argument("--skill",type=int,choices=range(1,6),default=1)
    parser.add_argument("--probe",action="store_true")
    parser.add_argument("--labels",action="store_true",help="Use demo-only two-letter labels instead of numeric indexes")
    parser.add_argument("--model",default=MODEL)
    parser.add_argument("--revision",default=None)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--local-files-only",action="store_true")
    args=parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or round(args.seconds * FPS) < 1:
        parser.error("--seconds must be finite and cover at least one frame")
    args.output.mkdir(parents=True,exist_ok=False)
    frames=args.output/"frames"
    frames.mkdir()
    torch.set_num_threads(4)
    game=create_game(args.seed,args.seconds,args.level,args.skill,args.output/"_vizdoom.ini")
    wad=Path(vzd.__file__).parent/"freedoom2.wad"
    nav=CampaignMap(wad,args.level,args.skill)
    if args.probe:
        try:
            inv=start_episode(game,args.skill)
            target=next(p for p in nav.items if p["kind"]=="Exit")
            obs=observe(game,game.get_state(),nav,"Reach exit",target,{"positions":[]})
            Image.fromarray(game.get_state().screen_buffer).save(args.output/"probe.jpg")
            image=Image.fromarray(np.uint8(nav.grid>=0)*200).convert("RGB").resize((nav.grid.shape[1]*4,nav.grid.shape[0]*4))
            draw=ImageDraw.Draw(image)
            for x,y in nav.last_path:
                px,py=(x-nav.x0)/8,(y-nav.y0)/8
                draw.ellipse((px-2,py-2,px+2,py+2),fill="yellow")
            for item in nav.items:
                px,py=(item["x"]-nav.x0)/8,(item["y"]-nav.y0)/8
                draw.ellipse((px-2,py-2,px+2,py+2),fill="red" if item["kind"]=="Exit" else "green")
            image.save(args.output/"map.png")
            report={"inventory":inv,"observation":obs,"map_items":nav.items,"navigable_cells":len(nav.valid),
                    "visible_labels":[p.object_name for p in game.get_state().labels]}
            (args.output/"probe.json").write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2))
        finally:
            game.close()
        return
    engine_class=LabelSystemOne if args.labels else SystemOne
    try:
        engine=engine_class.from_pretrained(args.model,revision=args.revision,
            model_kwargs={"torch_dtype":torch.bfloat16,"device_map":args.device,"attn_implementation":"sdpa",
                          "local_files_only":args.local_files_only},
            tokenizer_kwargs={"local_files_only":args.local_files_only})
    except BaseException:
        game.close()
        raise
    label_map=getattr(engine,"label_map",None)
    recorded_questions=label_questions(QUESTIONS,label_map) if args.labels else QUESTIONS
    metadata={
        "fps":FPS,"width":640,"height":480,"model":args.model,"model_revision":getattr(engine.model.config,"_commit_hash",None),
        "gamelevel":args.level,"scenario":f"Freedoom 2 campaign {args.level}","seed":args.seed,"skill":args.skill,
        "standing_order":STANDING_ORDER,
        "questions":{k:{"instructions":q.instructions,"criteria":q.criteria} for k,q in recorded_questions.items()},
        "buttons":[b.name for b in BUTTONS],"precision":"bfloat16",
        "gpu":torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "label_encoding":"two-letter" if args.labels else "numeric",
        "torch":torch.__version__,"transformers":transformers.__version__,"vizdoom":vzd.__version__,
        "observation_source":"Visible actor labels and game variables, plus actual WAD item/exit coordinates and collision geometry. Grid A* supplies a route waypoint bearing; it never selects controls. No image input to the model.",
        "control_method":"Seven control heads per SystemOne call, including independent navigation strafe. Emergency dodge takes priority over strafe. Planning uses a goal call then a target call conditioned on that new goal. Single worker; inference core unchanged. Every input button and turn direction selected by model; no aim/fire/navigation override.",
        "temporal_dependency":"Initial plan, then plan after each N applied completed control inferences. Both goal and target commit together; the next control request observes that committed plan. Previous buttons remain held during planning.",
        "plan_every":args.plan_every,"cadence_unit":"applied_completed_control_inferences",
        "schema_version":2,"planning_heads":list(PLAN_HEADS),"control_heads":list(CONTROL_HEADS),
        "cache_prefix":args.cache_prefix,
        "forward_passes_per_model_call":{"goal":1,"target":1,"control":2 if args.cache_prefix else 1},
        "forward_passes_per_plan":2,
        "model_calls_per_plan":2,"model_calls_per_control":1,
        "timing":"35 Hz game clock is independent of plan cadence; no planning timer. Model latencies time system_one calls; planning latency includes both stages and prompt preparation. Control gaps measure actual applications including intervening planning and polling.",
        "loadout_commands":["give shotgun","take Shell 999",f"give Shell {1 if args.skill==1 else 2}","take Clip 999",f"give Clip {15 if args.skill==1 else 30}"],
        "inventory_policy":"Loadout initialized only at recorded episode starts. No refills, invulnerability, teleports, enemy respawns or scripted actions.",
        "capture_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "library_sha256":hashlib.sha256(Path(__import__("system_one.inference",fromlist=["x"]).__file__).read_bytes()).hexdigest(),
        "assets_sha256":hashlib.sha256(wad.read_bytes()).hexdigest(),"audio_sample_rate":44100,
    }
    if args.labels:
        (args.output/"labelmap.json").write_text(json.dumps(engine.label_validation,indent=2))
    try:
        initial=start_episode(game,args.skill)
        metadata["startinginventory"]=initial
        goal="Reach exit"
        target=next(p for p in nav.items if p["kind"]=="Exit")
        memory={"positions":[]}
        obs=observe(game,game.get_state(),nav,goal,target,memory)
        print("MAP",len(nav.valid),"cells",len(nav.items),"items; starting",initial,flush=True)
        warmups=[]
        for _ in range(2):
            warm_request={"kind":"plan","episode":0,"plan_id":0,"frame":0,"observation":obs}
            warm_plan=infer_request(engine,warm_request,args.cache_prefix)
            warm_obs=observe(game,game.get_state(),nav,warm_plan["active_goal"],warm_plan["active_target"],memory)
            warm_control=infer_request(engine,dict(warm_request,kind="control",observation=warm_obs),args.cache_prefix)
            warmups.append({"plan":warm_plan,"control":warm_control})
        metadata["warmup_model_ms"]=sum(r["latency_ms"] for warmup in warmups for r in warmup.values())
        memory={"positions":[]}
        (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2))
        print("MODEL READY; recording",flush=True)
        with ThreadPoolExecutor(max_workers=1) as worker,(args.output/"decisions.jsonl").open("w",buffering=1) as decisions, \
                (args.output/"events.jsonl").open("w",buffering=1) as events,wave.open(str(args.output/"audio.wav"),"wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(44100)
            action=[0]*len(BUTTONS)
            future=None
            request=None
            episode=1
            cadence=PlanCadence(args.plan_every)
            batch=0
            latencies=[]
            plan_latencies=[]
            goal_latencies=[]
            target_latencies=[]
            control_gaps=[]
            model_ms=0.
            discarded_model_ms=0.
            plan_count=0
            controls_plan_id=None
            answer_frames={}
            total_kills=0
            last_kills=0
            late=0
            resets=[]
            events.write(json.dumps({"frame":0,"event":"episode_start","episode":1,"inventory":initial})+"\n")
            started=time.perf_counter()
            for frame in range(round(args.seconds*FPS)):
                if game.is_episode_finished() or game.is_player_dead():
                    reason="death" if game.is_player_dead() else "level_finished_or_timeout"
                    if future:
                        discarded=future.result()
                        discarded_model_ms+=discarded["latency_ms"]
                        events.write(json.dumps({"frame":frame,"event":"discarded_episode_boundary_inference","result":discarded})+"\n")
                        future=None
                    total_kills+=last_kills
                    last_kills=0
                    episode+=1
                    cadence.reset(episode)
                    controls_plan_id=None
                    answer_frames={}
                    inv=start_episode(game,args.skill)
                    nav=CampaignMap(wad,args.level,args.skill)
                    goal="Reach exit"
                    target=next(p for p in nav.items if p["kind"]=="Exit")
                    memory={"positions":[]}
                    action=[0]*len(BUTTONS)
                    event={"frame":frame,"event":"episode_reset","reason":reason,"episode":episode,"inventory":inv}
                    resets.append(event)
                    events.write(json.dumps(event)+"\n")
                    decisions.write(json.dumps({"frame":frame,"batch":batch,"kind":"reset","episode":episode,
                        "latency_ms":0,"answers":{},"evaluated_heads":[],"plan_updated":False,
                        "plan_id":cadence.plan_id,"controls_since_plan":0,"action":action})+"\n")
                state=game.get_state()
                completed=None
                if future and future.done():
                    completed=future.result()
                    future=None
                    if not cadence.accepts(completed):
                        discarded_model_ms+=completed["latency_ms"]
                        events.write(json.dumps({"frame":frame,"event":"discarded_stale_inference","result":completed})+"\n")
                        completed=None
                    elif completed["kind"]=="control":
                        action=controls(completed["answers"])
                    else:
                        goal=completed["active_goal"]
                        target=completed["active_target"]
                if future is None and completed is None:
                    obs=observe(game,state,nav,goal,target,memory)
                    request={"kind":cadence.next_kind,"episode":episode,"plan_id":cadence.plan_id,
                             "frame":frame,"observation":obs,"submitted":time.perf_counter()}
                    future=worker.submit(infer_request,engine,request,args.cache_prefix)
                last_kills=int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
                inv=inventory(game)
                Image.fromarray(state.screen_buffer).save(frames/f"{frame:06d}.jpg",quality=92)
                audio.writeframesraw(state.audio_buffer.tobytes())
                applied_at=time.perf_counter()
                game.make_action(action,1)
                if completed is not None:
                    control_gap_ms=(applied_at-cadence.last_control_applied)*1000 if (
                        completed["kind"]=="control" and cadence.last_control_applied is not None) else None
                    control_gap_frames=frame-cadence.last_control_frame if control_gap_ms is not None else None
                    cadence.commit(completed,applied_at,frame)
                    batch+=1
                    model_ms+=completed["latency_ms"]
                    if completed["kind"]=="control":
                        controls_plan_id=cadence.plan_id
                        latencies.append(completed["control_latency_ms"])
                        if control_gap_ms is not None:
                            control_gaps.append(control_gap_ms)
                    else:
                        plan_count+=1
                        plan_latencies.append(completed["planning_latency_ms"])
                        goal_latencies.append(completed["goal_latency_ms"])
                        target_latencies.append(completed["target_latency_ms"])
                    answer_frames.update({head:request["frame"] for head in completed["evaluated_heads"]})
                    row={**completed,"frame":frame,"batch":batch,"episode":episode,
                         "plan_id":cadence.plan_id,"plan_updated":completed["kind"]=="plan",
                         "controls_since_plan":cadence.controls_since_plan,"controls_plan_id":controls_plan_id,
                         "answers":dict(cadence.answers),"questions":dict(cadence.questions),
                         "answer_observation_frames":dict(answer_frames),
                         "action":action,"selected_controls":dict(zip(metadata["buttons"],action)),
                         "observation":request["observation"],"active_goal":goal,"active_target":target,
                         "request_wall_seconds":request["submitted"]-started,
                         "applied_wall_seconds":applied_at-started,
                         "request_to_apply_ms":(applied_at-request["submitted"])*1000,
                         "completion_to_apply_ms":(applied_at-completed["worker_completed"])*1000,
                         "control_gap_ms":control_gap_ms,"control_gap_frames":control_gap_frames,
                         "health":inv["health"],"ammo":inv["selected_weapon_ammo"],
                         "kills":total_kills+last_kills}
                    decisions.write(json.dumps(row)+"\n")
                    if completed["kind"]=="plan":
                        events.write(json.dumps({"frame":frame,"event":"plan_committed","episode":episode,
                            "plan_id":cadence.plan_id,"goal":goal,"target":target,
                            "controls_since_plan":0,"result":completed})+"\n")
                    if batch%20==0:
                        print(f"frame={frame} batch={batch} kind={completed['kind']} plan={cadence.plan_id} "
                              f"controls={cadence.controls_since_plan} health={inv['health']} "
                              f"latency={completed['latency_ms']:.0f}",flush=True)
                    # Submit from a fresh main-thread snapshot, without waiting another 35 Hz tick.
                    if not game.is_episode_finished() and not game.is_player_dead() and frame+1<round(args.seconds*FPS):
                        obs=observe(game,game.get_state(),nav,goal,target,memory)
                        request={"kind":cadence.next_kind,"episode":episode,"plan_id":cadence.plan_id,
                                 "frame":frame+1,"observation":obs,"submitted":time.perf_counter()}
                        future=worker.submit(infer_request,engine,request,args.cache_prefix)
                remaining=started+(frame+1)/FPS-time.perf_counter()
                if remaining>0:
                    time.sleep(remaining)
                else:
                    late+=1
            elapsed=time.perf_counter()-started
            if future:
                pending=future.result()
                discarded_model_ms+=pending["latency_ms"]
                events.write(json.dumps({"frame":frame+1,"event":"unapplied_final_inference","result":pending})+"\n")
        metadata.update(total_frames=frame+1,duration_seconds=(frame+1)/FPS,capture_wall_seconds=elapsed,
                        late_frames=late,decisions=batch,control_updates=len(latencies),plans=plan_count,
                        episodes=episode,resets=resets,kills=total_kills+last_kills,
                        median_latency_ms=float(np.median(latencies)) if latencies else None,
                        p95_latency_ms=float(np.percentile(latencies,95)) if latencies else None,
                        median_control_latency_ms=float(np.median(latencies)) if latencies else None,
                        p95_control_latency_ms=float(np.percentile(latencies,95)) if latencies else None,
                        actual_decisions_per_second=batch/elapsed,actual_controls_per_second=len(latencies)/elapsed,
                        applied_model_ms=model_ms,unapplied_model_ms=discarded_model_ms,
                        recording_model_ms=model_ms+discarded_model_ms,
                        planning_wall_ms=sum(plan_latencies),goal_model_ms=sum(goal_latencies),
                        target_model_ms=sum(target_latencies),control_model_ms=sum(latencies),
                        median_planning_latency_ms=float(np.median(plan_latencies)) if plan_latencies else None,
                        median_goal_latency_ms=float(np.median(goal_latencies)) if goal_latencies else None,
                        median_target_latency_ms=float(np.median(target_latencies)) if target_latencies else None,
                        median_control_gap_ms=float(np.median(control_gaps)) if control_gaps else None,
                        p95_control_gap_ms=float(np.percentile(control_gaps,95)) if control_gaps else None,
                        max_control_gap_ms=max(control_gaps) if control_gaps else None,final_inventory=inventory(game),
                        explored_cells=len(nav.visited),collected_or_passed=sorted(nav.collected))
        (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2))
        print("COMPLETE",json.dumps({k:v for k,v in metadata.items() if k not in {"questions","buttons"}}),flush=True)
    finally:
        game.close()


if __name__=="__main__":
    main()

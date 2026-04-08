import sys, numpy as np, importlib.util, os
sys.path.insert(0, '/home/terror86/spe/arc3/games')
from arcengine import ActionInput, GameAction, GameState, ARCBaseGame
from collections import deque
from scipy import ndimage

ACTIONS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4]
DEFAULTS={GameAction.ACTION1:(0,-1),GameAction.ACTION2:(0,1),
          GameAction.ACTION3:(-1,0),GameAction.ACTION4:(1,0)}

def find_all(lg,color):
    if color is None or color not in set(lg.flatten().tolist()): return []
    return [(int(c),int(r)) for r,c in zip(*np.where(lg==color))]

def bfs_path(lg,start,goal,block,amap):
    if not start or not goal: return []
    h,w=lg.shape; vis={start}; q=deque([(start,[])])
    while q:
        (x,y),path=q.popleft()
        if (x,y)==goal: return path
        for a,(dx,dy) in amap.items():
            nx,ny=x+dx,y+dy
            if not(0<=nx<w and 0<=ny<h): continue
            if (nx,ny) in vis: continue
            if int(lg[ny,nx]) in block: continue
            vis.add((nx,ny)); q.append(((nx,ny),path+[a]))
    return []

def _get_clean_deltas(g0,g1,scale,color):
    """Get movement deltas using nearest-neighbor matching. Only return clean ±1 deltas."""
    lg0=g0[::scale,::scale]; lg1=g1[::scale,::scale]
    if color not in set(lg0.flatten().tolist()): return []
    lab0,n0=ndimage.label(lg0==color)
    lab1,_=ndimage.label(lg1==color)
    comps0=[]; comps1=[]
    for i in range(1,n0+1):
        r,c=np.where(lab0==i)
        if len(r): comps0.append((float(np.mean(c)),float(np.mean(r)),len(r)))
    ncomp=int(lab1.max())
    for j in range(1,ncomp+1):
        r,c=np.where(lab1==j)
        if len(r): comps1.append((float(np.mean(c)),float(np.mean(r)),len(r)))
    deltas=[]
    for cx0,cy0,sz in comps0:
        best=None; best_d=999
        for cx1,cy1,sz1 in comps1:
            if sz1!=sz: continue
            d=abs(cx1-cx0)+abs(cy1-cy0)
            if d<best_d: best_d=d; best=(cx1-cx0,cy1-cy0)
        if best:
            dx,dy=best
            # Only accept clean unit deltas
            if (abs(round(dx))==1 and abs(round(dy))==0) or                (abs(round(dx))==0 and abs(round(dy))==1):
                deltas.append((int(round(dx)),int(round(dy))))
    return deltas

def find_scale_and_player(g0,g1):
    best=None; best_score=999
    for scale in [1,2,4,8]:
        for color in set(g0[::scale,::scale].flatten().tolist())-{0}:
            deltas=_get_clean_deltas(g0,g1,scale,color)
            if not deltas: continue
            dx,dy=deltas[0]
            # Score: prefer smaller objects (lower count = more specific)
            count=int(np.sum(g0[::scale,::scale]==color))
            score=count/100
            if score<best_score:
                best_score=score
                best={'scale':scale,'player':color,'dx':dx,'dy':dy}
    return best

class GameModel:
    def __init__(self):
        self.scale=None; self.player=None; self.goal=None
        self.walls=set(); self.floors=set(); self.deadly=set()
        self.pickups=set(); self.amap={}
        self.block_goal_until_clear=False; self.win_levels=None

    def sc(self): return self.scale or 8
    def amap_use(self): return self.amap if len(self.amap)>=2 else DEFAULTS

    def block_set(self,extra=frozenset()):
        b=set(self.walls)|set(self.deadly)|set(extra)
        b-=set(self.floors)|{0}
        if self.goal is not None: b.discard(self.goal)
        if self.player is not None: b.discard(self.player)
        return b

    def infer_goal(self,lg):
        if self.goal is not None: return
        known=set(self.walls)|set(self.floors)|set(self.deadly)|set(self.pickups)|{0}
        if self.player is not None: known.add(self.player)
        cands=set(lg.flatten().tolist())-known
        if not cands: return
        counts={c:int(np.sum(lg==c)) for c in cands}
        self.goal=min(counts,key=counts.get)
        self.walls.discard(self.goal)

    def _clean(self):
        self.walls-=set(self.floors)
        if self.goal is not None: self.walls.discard(self.goal); self.pickups.discard(self.goal)
        if self.player is not None: self.walls.discard(self.player); self.pickups.discard(self.player)
        self.pickups-=set(self.walls)|set(self.floors)|set(self.deadly)

    def learn(self,action,g0,g1,state,lv0,lv1,win_levels):
        if win_levels: self.win_levels=win_levels
        diff=(g0!=g1)
        if not diff.any():
            if self.player is not None:
                sc=self.sc(); lg0=g0[::sc,::sc]
                for pr,pc2 in zip(*np.where(lg0==self.player)):
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=int(pr)+dr,int(pc2)+dc
                        if 0<=nr<lg0.shape[0] and 0<=nc<lg0.shape[1]:
                            n=int(lg0[nr,nc])
                            if n not in set(self.floors)|{0,self.player,self.goal}:
                                self.walls.add(n)
            return

        if self.scale is None or self.player is None:
            r=find_scale_and_player(g0,g1)
            if r:
                self.scale=r['scale']; self.player=r['player']
                if (r['dx'],r['dy'])!=(0,0) and action not in self.amap:
                    self.amap[action]=(r['dx'],r['dy'])
        else:
            sc=self.sc()
            deltas=_get_clean_deltas(g0,g1,sc,self.player)
            if deltas and action not in self.amap:
                self.amap[action]=deltas[0]

        if self.player is not None:
            sc=self.sc(); lg0=g0[::sc,::sc]; lg1=g1[::sc,::sc]
            pos0=set(map(tuple,zip(*np.where(lg0==self.player))))
            pos1=set(map(tuple,zip(*np.where(lg1==self.player))))
            for r,c in pos0-pos1:
                fc=int(lg1[r,c])
                if fc!=self.player: self.floors.add(fc)

        sc=self.sc(); lg0=g0[::sc,::sc]; lg1=g1[::sc,::sc]
        if lv1>lv0 and self.player is not None:
            for pr,pc2 in zip(*np.where(lg0==self.player)):
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(0,0)]:
                    nr,nc=int(pr)+dr,int(pc2)+dc
                    if 0<=nr<lg0.shape[0] and 0<=nc<lg0.shape[1]:
                        n=int(lg0[nr,nc])
                        if n not in {0,self.player}|self.walls|self.floors|self.pickups|self.deadly:
                            if self.goal is None: self.goal=n; self.walls.discard(n)
                            return

        if state==GameState.GAME_OVER and self.player is not None:
            for pr,pc2 in zip(*np.where(lg0==self.player)):
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=int(pr)+dr,int(pc2)+dc
                    if 0<=nr<lg0.shape[0] and 0<=nc<lg0.shape[1]:
                        n=int(lg0[nr,nc])
                        if n not in {0,self.player}|self.walls|self.floors:
                            self.deadly.add(n)
        self._clean()

    def is_won(self,grid,state):
        if state==GameState.WIN: return True
        sc=self.sc(); lg=grid[::sc,::sc]
        if self.player is not None and self.player not in set(lg.flatten().tolist()):
            if state!=GameState.GAME_OVER: return True
        return False

    def plan(self,grid):
        sc=self.sc(); lg=grid[::sc,::sc]
        self.infer_goal(lg)
        if self.player is None: return []
        players=find_all(lg,self.player)
        if not players: return []
        goals=find_all(lg,self.goal) if self.goal is not None else []
        pickups=[]
        for c in self.pickups: pickups.extend(find_all(lg,c))
        amap=self.amap_use()
        targets=pickups if pickups else goals
        if not targets: return []
        blk=self.block_set(frozenset(goals)
            if pickups and self.block_goal_until_clear else frozenset())
        # Find shortest path from ANY player piece to ANY target
        # But ONLY count paths where the first action actually moves the piece
        best=None
        for pp in players:
            for t in targets:
                p=bfs_path(lg,pp,t,blk,amap)
                if not p: continue
                # Verify first action actually moves this piece (not a 0-step path)
                if len(p)==0: continue
                dx,dy=amap.get(p[0],(0,0))
                nx,ny=pp[0]+dx,pp[1]+dy
                if not(0<=nx<lg.shape[1] and 0<=ny<lg.shape[0]): continue
                if int(lg[ny,nx]) in blk: continue
                if (best is None or len(p)<len(best)): best=p
        return best or []

    def player_pos(self,grid):
        pp=find_all(grid[::self.sc(),::self.sc()],self.player)
        return pp[0] if pp else None

    def summary(self):
        return (f"player={self.player} goal={self.goal} sc={self.scale} "
                f"walls={self.walls} floors={self.floors} amap={len(self.amap)}")

class SPEAgent:
    PROBE=([GameAction.ACTION4]*4+[GameAction.ACTION2]*4+
           [GameAction.ACTION3]*4+[GameAction.ACTION1]*4+
           [GameAction.ACTION4,GameAction.ACTION1,
            GameAction.ACTION3,GameAction.ACTION2]*3)

    def __init__(self):
        self.model=GameModel(); self.path=[]
        self.prev=None; self.prev_lv=0; self.prev_pp=None
        self.step=0; self.probe_i=0
        self.last=GameAction.ACTION4; self.stuck=0

    def teleported(self,grid):
        if self.model.player is None: return False
        np2=self.model.player_pos(grid)
        if np2 is not None and self.prev_pp is not None:
            if abs(np2[0]-self.prev_pp[0])+abs(np2[1]-self.prev_pp[1])>3: return True
        return False

    def choose(self,grid,lv,state,wl):
        self.step+=1
        if self.prev is not None:
            self.model.learn(self.last,self.prev,grid,state,self.prev_lv,lv,wl)
        if self.probe_i<len(self.PROBE):
            act=self.PROBE[self.probe_i]; self.probe_i+=1
            if self.probe_i==len(self.PROBE): self.path=[]
        else:
            tp=self.teleported(grid)
            if lv!=self.prev_lv or tp or not self.path or self.stuck>=15:
                self.path=self.model.plan(grid); self.stuck=0
            act=self.path.pop(0) if self.path else ACTIONS[self.step%4]
            self.stuck=self.stuck+1 if not self.path else 0
        self.prev_pp=self.model.player_pos(grid)
        self.last=act; self.prev=grid.copy(); self.prev_lv=lv
        return act

def load_cls(fname):
    gname=fname.replace('.py','')
    spec=importlib.util.spec_from_file_location(gname,
        f'/home/terror86/spe/arc3/games/{fname}')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return next((getattr(mod,a) for a in dir(mod)
                 if isinstance(getattr(mod,a),type)
                 and issubclass(getattr(mod,a),ARCBaseGame)
                 and getattr(mod,a)!=ARCBaseGame),None)

def run(cls,max_steps=500,verbose=True):
    game=cls(); agent=SPEAgent()
    frame=game.perform_action(ActionInput(id=GameAction.RESET),raw=True)
    grid=np.array(frame.frame)[0]
    for step in range(max_steps):
        if agent.model.is_won(grid,frame.state):
            if verbose: print(f"    WIN at step {step}!")
            return "WIN",frame.levels_completed,step,agent.model
        if frame.state==GameState.GAME_OVER:
            agent.model.learn(agent.last,agent.prev if agent.prev is not None else grid,
                              grid,GameState.GAME_OVER,agent.prev_lv,
                              frame.levels_completed,frame.win_levels)
            frame=game.perform_action(ActionInput(id=GameAction.RESET),raw=True)
            grid=np.array(frame.frame)[0]; agent.path=[]; continue
        act=agent.choose(grid,frame.levels_completed,frame.state,frame.win_levels)
        prev_lv=frame.levels_completed
        frame=game.perform_action(ActionInput(id=act),raw=True)
        grid=np.array(frame.frame)[0]
        if frame.levels_completed>prev_lv:
            if verbose: print(f"    Step {step+1}: Level {frame.levels_completed}!")
    return "TIMEOUT",frame.levels_completed,max_steps,agent.model

print("="*55)
print("  SPE BLIND AGENT v17")
print("="*55)
wins=0; total=0
games_dir='/home/terror86/spe/arc3/games'
for fname in sorted(os.listdir(games_dir)):
    if not fname.endswith('.py') or fname=='main.py': continue
    cls=load_cls(fname)
    if not cls: continue
    total+=1; gname=fname.replace('.py','')
    print(f"\n{gname}:")
    try:
        s,lv,st,model=run(cls)
        icon='✓' if s=='WIN' else '✗'
        print(f"  {icon} {s} levels={lv} steps={st}")
        print(f"  Model: {model.summary()}")
        if s=='WIN': wins+=1
    except Exception as e:
        import traceback; traceback.print_exc(); total-=1
print(f"\n{'='*55}\n  Wins: {wins}/{total}\n{'='*55}")
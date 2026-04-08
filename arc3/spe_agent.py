
"""
SPE ARC-AGI-3 Agent
Sovereign Perception Engine — Interactive Game Solver

Architecture:
- Frame parser: 64x64 grid → logical objects
- Causal model: action → state change mapping  
- BFS planner: find path to goal
- Hypothesis tracker: learn game mechanics from observation
"""
import sys, numpy as np, os, importlib.util
sys.path.insert(0, "/home/terror86/spe/arc3/games")
from collections import deque, defaultdict
from arcengine import ActionInput, GameAction, GameState, ARCBaseGame

ACTION_DELTAS = {
    GameAction.ACTION1: (0,-1),  # up
    GameAction.ACTION2: (0, 1),  # down
    GameAction.ACTION3: (-1,0),  # left
    GameAction.ACTION4: (1, 0),  # right
}

class FrameParser:
    """Parse 64x64 frames into logical game state."""
    
    def __init__(self):
        self.scale = 8
        self.wall_colors = set()
        self.player_color = None
        self.goal_color = None
        
    def detect_scale(self, grid64):
        """Detect upscaling factor from sprite block size."""
        for color in [8,9,2,1,3,4,6,7,11,14]:
            rows=np.where(np.any(grid64==color,axis=1))[0]
            if len(rows)>=2:
                # Find first contiguous block
                for i in range(len(rows)-1):
                    if rows[i+1]==rows[i]+1:
                        # Find block size
                        start=rows[i]
                        size=1
                        while i+size<len(rows) and rows[i+size]==rows[i]+size:
                            size+=1
                        if size>=1: return size
        return 8
    
    def get_logical(self, grid64):
        self.scale = self.detect_scale(grid64)
        return grid64[::self.scale, ::self.scale]
    
    def get_color_info(self, lg):
        """Return color→count and color→positions."""
        colors = set(lg.flatten().tolist()) - {0}
        counts = {c:int(np.sum(lg==c)) for c in colors}
        positions = {}
        for c in colors:
            pos = list(zip(*np.where(lg==c)))
            positions[c] = [(int(r),int(col)) for r,col in pos]
        return counts, positions
    
    def identify_roles(self, lg, prev_lg=None):
        """Identify wall, player, goal colors."""
        counts, positions = self.get_color_info(lg)
        if not counts: return
        
        sorted_colors = sorted(counts.items(), key=lambda x: x[1])
        
        # Largest = wall
        if sorted_colors:
            self.wall_colors = {sorted_colors[-1][0]}
        
        movable = [(c,n) for c,n in sorted_colors if c not in self.wall_colors]
        
        # Moving object = player
        if prev_lg is not None and movable:
            for c,n in movable:
                prev_pos = set(zip(*np.where(prev_lg==c))) if c in set(prev_lg.flatten()) else set()
                curr_pos = set(zip(*np.where(lg==c)))
                if prev_pos != curr_pos:
                    self.player_color = c
                    break
        
        if self.player_color is None and movable:
            self.player_color = movable[0][0]
        
        if self.goal_color is None and len(movable)>=2:
            goal_candidates = [c for c,n in movable if c!=self.player_color]
            if goal_candidates:
                self.goal_color = goal_candidates[0]


class CausalHypothesis:
    """
    Track causal model: what does each action do?
    Learn from observation: action A in state S → state S'
    """
    def __init__(self):
        self.action_effects = defaultdict(list)  # action → list of (before,after) diffs
        self.stuck_count = 0
        self.action_sequence = []
        
    def record(self, action, before_grid, after_grid):
        diff = (before_grid != after_grid)
        if diff.any():
            self.stuck_count = 0
        else:
            self.stuck_count += 1
        self.action_effects[action].append(diff.sum())
        self.action_sequence.append(action)
        
    def is_stuck(self):
        return self.stuck_count >= 4
    
    def least_tried_action(self, exclude=None):
        counts = {a:len(self.action_effects[a]) for a in ACTION_DELTAS}
        if exclude: counts = {a:c for a,c in counts.items() if a not in exclude}
        return min(counts, key=counts.get)


class SPEGameAgent:
    """
    Full SPE agent for one game session.
    Combines frame parsing, causal reasoning, and BFS planning.
    """
    def __init__(self):
        self.parser = FrameParser()
        self.hypothesis = CausalHypothesis()
        self.prev_grid = None
        self.step = 0
        self.visited_states = set()
        
    def bfs(self, lg, start, goal):
        if not start or not goal: return []
        sx,sy = start; gx,gy = goal
        h,w = lg.shape
        vis={(sx,sy)}; q=deque([(sx,sy,[])])
        while q:
            x,y,path = q.popleft()
            if x==gx and y==gy: return path
            for a,(dx,dy) in ACTION_DELTAS.items():
                nx,ny = x+dx,y+dy
                if not(0<=nx<w and 0<=ny<h): continue
                if (nx,ny) in vis: continue
                if int(lg[ny,nx]) in self.parser.wall_colors: continue
                vis.add((nx,ny)); q.append((nx,ny,path+[a]))
        return []
    
    def find_pos(self, lg, color):
        if color is None: return None
        pos=list(zip(*np.where(lg==color)))
        if not pos: return None
        r,c=pos[0]; return (int(c),int(r))
    
    def choose_action(self, grid64):
        prev_lg = self.parser.get_logical(self.prev_grid) if self.prev_grid is not None else None
        lg = self.parser.get_logical(grid64)
        self.parser.identify_roles(lg, prev_lg)
        
        player = self.find_pos(lg, self.parser.player_color)
        goal   = self.find_pos(lg, self.parser.goal_color)
        
        # Try BFS to goal
        path = self.bfs(lg, player, goal)
        if path:
            return path[0]
        
        # If stuck, try exploration
        if self.hypothesis.is_stuck():
            return self.hypothesis.least_tried_action()
        
        # Try systematic movement
        actions = [GameAction.ACTION1,GameAction.ACTION2,
                   GameAction.ACTION3,GameAction.ACTION4]
        return actions[self.step % 4]
    
    def update(self, action, grid64):
        if self.prev_grid is not None:
            self.hypothesis.record(action, self.prev_grid, grid64)
        self.prev_grid = grid64.copy()
        self.step += 1


def load_game_class(filepath):
    """Dynamically load a game class from file."""
    name = os.path.basename(filepath).replace(".py","")
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj,type) and issubclass(obj,ARCBaseGame) and obj!=ARCBaseGame:
            return obj
    return None


def run_spe_agent(game_class, max_steps=300, verbose=True):
    """Run SPE agent on a game. Returns (status, levels_completed, steps)."""
    game  = game_class()
    agent = SPEGameAgent()
    frame = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    grid  = np.array(frame.frame)[0]
    
    for step in range(max_steps):
        if frame.state in [GameState.WIN, GameState.GAME_OVER]: break
        
        action = agent.choose_action(grid)
        prev_lv = frame.levels_completed
        frame   = game.perform_action(ActionInput(id=action), raw=True)
        new_grid = np.array(frame.frame)[0]
        agent.update(action, new_grid)
        grid = new_grid
        
        if frame.levels_completed > prev_lv:
            if verbose: print(f"    Step {step+1}: Level {frame.levels_completed}!")
            # Reset agent state for new level
            agent.prev_grid = None
            
    status = ("WIN" if frame.state==GameState.WIN else
              "OVER" if frame.state==GameState.GAME_OVER else "TIMEOUT")
    return status, frame.levels_completed, step+1


if __name__ == "__main__":
    games_dir = "/home/terror86/spe/arc3/games"
    results = {}
    
    print("="*55)
    print("  SPE ARC-AGI-3 AGENT — LOCAL GAMES")
    print("="*55)
    
    for fname in sorted(os.listdir(games_dir)):
        if not fname.endswith(".py") or fname=="main.py": continue
        gname = fname.replace(".py","")
        cls = load_game_class(f"{games_dir}/{fname}")
        if not cls:
            print(f"  {gname}: no class"); continue
        print(f"\n  {gname}:")
        try:
            status,levels,steps = run_spe_agent(cls)
            results[gname] = (status,levels,steps)
            icon = "✓" if status=="WIN" else "✗"
            print(f"    {icon} {status} levels={levels} steps={steps}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[gname] = ("ERROR",0,0)
    
    wins = sum(1 for s,l,st in results.values() if s=="WIN")
    print(f"\n  Won: {wins}/{len(results)} games")
    print("="*55)

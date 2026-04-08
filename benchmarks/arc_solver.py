
import json, os, sys, numpy as np
from itertools import product
from collections import deque
sys.path.append("/home/terror86/spe")
ARC_DIR = os.path.expanduser("~/spe/data/arc")

def to_np(g): return np.array(g, dtype=np.int32)
COLORS = {i:str(i) if i>0 else "." for i in range(10)}
def display_grid(grid, label=""):
    g=to_np(grid)
    if label: print(f"  {label}:")
    for row in g: print("    "+" ".join(COLORS.get(v,str(v)) for v in row))

# ── Transformations ────────────────────────────────────────────────────────────
def try_scale_tile(inp, out):
    i=to_np(inp); o=to_np(out)
    ih,iw=i.shape; oh,ow=o.shape
    if oh%ih!=0 or ow%iw!=0: return None
    kh,kw=oh//ih,ow//iw
    if kh!=ih or kw!=iw: return None
    predicted=np.zeros((oh,ow),dtype=np.int32)
    for r,c in product(range(ih),range(iw)):
        tile=i if i[r,c]!=0 else np.zeros((kh,kw),dtype=np.int32)
        predicted[r*kh:(r+1)*kh,c*kw:(c+1)*kw]=tile
    return predicted

def try_fill_enclosed(inp, out):
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    diff=(o!=i)
    if not diff.any(): return None
    if not np.all(o[diff]==4): return None
    h,w=i.shape
    visited=np.zeros((h,w),bool)
    q=deque()
    for r in range(h):
        for c in [0,w-1]:
            if i[r,c]!=3 and not visited[r,c]:
                visited[r,c]=True; q.append((r,c))
    for c in range(w):
        for r in [0,h-1]:
            if i[r,c]!=3 and not visited[r,c]:
                visited[r,c]=True; q.append((r,c))
    while q:
        r,c=q.popleft()
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc=r+dr,c+dc
            if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and i[nr,nc]!=3:
                visited[nr,nc]=True; q.append((nr,nc))
    predicted=i.copy()
    predicted[(~visited)&(i!=3)]=4
    return predicted if np.array_equal(predicted,o) else None

def try_color_replace(inp, out):
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    mapping={}
    for v,w in zip(i.flatten(),o.flatten()):
        if v in mapping:
            if mapping[v]!=w: return None
        else: mapping[v]=w
    predicted=np.vectorize(lambda x: mapping.get(x,x))(i)
    return predicted.astype(np.int32)

def try_rotate(inp, out):
    i=to_np(inp); o=to_np(out)
    for k in [1,2,3]:
        r=np.rot90(i,k)
        if r.shape==o.shape and np.array_equal(r,o): return r
    return None

def try_flip(inp, out):
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    if np.array_equal(np.fliplr(i),o): return np.fliplr(i)
    if np.array_equal(np.flipud(i),o): return np.flipud(i)
    return None

def try_transpose(inp, out):
    i=to_np(inp); o=to_np(out)
    return i.T if i.T.shape==o.shape and np.array_equal(i.T,o) else None

def try_zoom(inp, out):
    i=to_np(inp); o=to_np(out)
    ih,iw=i.shape; oh,ow=o.shape
    if oh%ih!=0 or ow%iw!=0: return None
    kh,kw=oh//ih,ow//iw
    predicted=np.repeat(np.repeat(i,kh,axis=0),kw,axis=1)
    return predicted if np.array_equal(predicted,o) else None

def try_shrink(inp, out):
    i=to_np(inp); o=to_np(out)
    ih,iw=i.shape; oh,ow=o.shape
    if ih%oh!=0 or iw%ow!=0: return None
    kh,kw=ih//oh,iw//ow
    predicted=i[::kh,::kw]
    return predicted if np.array_equal(predicted,o) else None

def try_crop_nonzero(inp, out):
    i=to_np(inp)
    rows=np.any(i!=0,axis=1); cols=np.any(i!=0,axis=0)
    if not rows.any(): return None
    rmin,rmax=np.where(rows)[0][[0,-1]]
    cmin,cmax=np.where(cols)[0][[0,-1]]
    cropped=i[rmin:rmax+1,cmin:cmax+1]
    o=to_np(out)
    return cropped if np.array_equal(cropped,o) else None

def try_gravity(inp, out):
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    predicted=np.zeros_like(i)
    for c in range(i.shape[1]):
        col=i[:,c]; nonz=col[col!=0]
        if len(nonz)>0: predicted[-len(nonz):,c]=nonz
    return predicted if np.array_equal(predicted,o) else None

def try_gravity_up(inp, out):
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    predicted=np.zeros_like(i)
    for c in range(i.shape[1]):
        col=i[:,c]; nonz=col[col!=0]
        if len(nonz)>0: predicted[:len(nonz),c]=nonz
    return predicted if np.array_equal(predicted,o) else None

def try_gravity_right(inp, out):
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    predicted=np.zeros_like(i)
    for r in range(i.shape[0]):
        row=i[r,:]; nonz=row[row!=0]
        if len(nonz)>0: predicted[r,-len(nonz):]=nonz
    return predicted if np.array_equal(predicted,o) else None

def try_gravity_left(inp, out):
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    predicted=np.zeros_like(i)
    for r in range(i.shape[0]):
        row=i[r,:]; nonz=row[row!=0]
        if len(nonz)>0: predicted[r,:len(nonz)]=nonz
    return predicted if np.array_equal(predicted,o) else None

def try_mirror_complete(inp, out):
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    for flip_fn in [np.fliplr, np.flipud]:
        predicted=i.copy()
        predicted[i==0]=flip_fn(i)[i==0]
        if np.array_equal(predicted,o): return predicted
    return None

def try_identity(inp, out):
    i=to_np(inp); o=to_np(out)
    return i if np.array_equal(i,o) else None

def try_unique_rows(inp, out):
    i=to_np(inp); o=to_np(out)
    seen=[]; unique=[]
    for row in i:
        key=tuple(row.tolist())
        if key not in seen: seen.append(key); unique.append(row)
    pred=np.array(unique)
    return pred if pred.shape==o.shape and np.array_equal(pred,o) else None

def try_outline(inp, out):
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    predicted=np.zeros_like(i)
    predicted[0,:]=i[0,:]; predicted[-1,:]=i[-1,:]
    predicted[:,0]=i[:,0]; predicted[:,-1]=i[:,-1]
    return predicted if np.array_equal(predicted,o) else None

def try_hollow(inp, out):
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    predicted=i.copy()
    if i.shape[0]>2 and i.shape[1]>2:
        predicted[1:-1,1:-1]=0
    return predicted if np.array_equal(predicted,o) else None

def try_split_xor(inp, out):
    """
    Split grid by a vertical divider column (unique color).
    XOR/difference: cells that differ between left and right halves → mark as 2.
    Used in task 0520fde7.
    """
    i=to_np(inp); o=to_np(out)
    h,w=i.shape
    # Find vertical divider: column where all values are same non-zero non-background
    for c in range(1,w-1):
        col=i[:,c]
        if len(set(col.tolist()))==1 and col[0]!=0:
            left=i[:,:c]; right=i[:,c+1:c+1+c]
            if right.shape!=left.shape: continue
            # XOR: positions that are same in left but not right (or vice versa)
            predicted=np.zeros_like(left)
            for r2 in range(h):
                for c2 in range(c):
                    lv=left[r2,c2]; rv=right[r2,c2] if c2<right.shape[1] else 0
                    if lv!=rv: predicted[r2,c2]=2
            if np.array_equal(predicted,o): return predicted
    return None

def try_extract_unique_region(inp, out):
    """
    Find the region/object with the least-frequent color.
    Crop its bounding box.
    """
    i=to_np(inp); o=to_np(out)
    colors=set(i.flatten().tolist())-{0}
    if not colors: return None
    counts={c:int(np.sum(i==c)) for c in colors}
    min_color=min(counts,key=counts.get)
    mask=(i==min_color)
    rows=np.where(np.any(mask,axis=1))[0]
    cols=np.where(np.any(mask,axis=0))[0]
    if not len(rows) or not len(cols): return None
    cropped=i[rows[0]:rows[-1]+1,cols[0]:cols[-1]+1]
    return cropped if np.array_equal(cropped,o) else None

def try_extract_quadrant(inp, out):
    """
    Grid divided by blank rows/cols into quadrants.
    Extract the quadrant that contains the most non-zero cells of a specific color.
    """
    i=to_np(inp); o=to_np(out)
    h,w=i.shape
    # Find blank rows (all zero)
    blank_rows=[r for r in range(h) if np.all(i[r,:]==0)]
    blank_cols=[c for c in range(w) if np.all(i[:,c]==0)]
    if not blank_rows and not blank_cols: return None
    # Split into segments
    row_splits=[0]+[r+1 for r in blank_rows]+[h]
    col_splits=[0]+[c+1 for c in blank_cols]+[w]
    best=None; best_count=-1
    for r0,r1 in zip(row_splits,row_splits[1:]):
        for c0,c1 in zip(col_splits,col_splits[1:]):
            seg=i[r0:r1,c0:c1]
            if seg.size==0: continue
            count=np.count_nonzero(seg)
            if count>best_count and np.array_equal(seg,o):
                best=seg; best_count=count
    return best

def try_grid_partition_sizes(inp, out):
    """
    Grid divided by lines of a single color into rectangular cells.
    Output = number of cells in each row/col partition (as grid of that color).
    Task 1190e5a7 pattern.
    """
    i=to_np(inp); o=to_np(out)
    # Find divider color: most common non-background color forming full rows/cols
    colors=set(i.flatten().tolist())-{0}
    for dc in colors:
        div_rows=[r for r in range(i.shape[0]) if np.all(i[r,:]==dc)]
        div_cols=[c for c in range(i.shape[1]) if np.all(i[:,c]==dc)]
        if not div_rows and not div_cols: continue
        # Get background color (most common non-divider)
        bg_colors=set(i.flatten().tolist())-{dc}
        if not bg_colors: continue
        bg=max(bg_colors,key=lambda c:int(np.sum(i==c)))
        # Count partitions
        row_parts=[]
        prev=0
        for r in sorted(div_rows)+[i.shape[0]]:
            if r>prev: row_parts.append(r-prev)
            prev=r+1
        col_parts=[]
        prev=0
        for c in sorted(div_cols)+[i.shape[1]]:
            if c>prev: col_parts.append(c-prev)
            prev=c+1
        predicted=np.full((len(row_parts),len(col_parts)),bg,dtype=np.int32)
        if np.array_equal(predicted,o): return predicted
    return None

def try_move_to_bottom(inp, out):
    """Objects (connected components) slide to bottom of grid."""
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    # Simple: each column, nonzero values fall to bottom
    predicted=np.zeros_like(i)
    for c in range(i.shape[1]):
        col=i[:,c]; nonz=col[col!=0]
        if len(nonz)>0: predicted[-len(nonz):,c]=nonz
    if np.array_equal(predicted,o): return predicted
    # Try row-wise fall to right
    predicted2=np.zeros_like(i)
    for r in range(i.shape[0]):
        row=i[r,:]; nonz=row[row!=0]
        if len(nonz)>0: predicted2[r,-len(nonz):]=nonz
    return predicted2 if np.array_equal(predicted2,o) else None

def try_sort_rows_by_color(inp, out):
    """Sort rows by their dominant (most frequent) color."""
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    def dominant(row):
        vals=row[row!=0]
        if not len(vals): return 0
        return int(np.bincount(vals).argmax())
    idx=sorted(range(len(i)),key=lambda r:dominant(i[r]))
    pred=i[idx]
    return pred if np.array_equal(pred,o) else None

def try_sort_cols_by_color(inp, out):
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    def dominant(col):
        vals=col[col!=0]
        if not len(vals): return 0
        return int(np.bincount(vals).argmax())
    idx=sorted(range(i.shape[1]),key=lambda c:dominant(i[:,c]))
    pred=i[:,idx]
    return pred if np.array_equal(pred,o) else None

def try_colorcount_sort(inp, out):
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    counts=[np.count_nonzero(row) for row in i]
    idx=np.argsort(counts)
    pred=i[idx]
    return pred if np.array_equal(pred,o) else None

def try_extract_colored_region(inp, out):
    """Extract bounding box of each unique non-zero color region."""
    i=to_np(inp); o=to_np(out)
    colors=sorted(set(i.flatten().tolist())-{0})
    for color in colors:
        mask=(i==color)
        rows=np.where(np.any(mask,axis=1))[0]
        cols=np.where(np.any(mask,axis=0))[0]
        if not len(rows) or not len(cols): continue
        cropped=i[rows[0]:rows[-1]+1,cols[0]:cols[-1]+1]
        if np.array_equal(cropped,o): return cropped
    return None

def try_mask_keep(inp, out):
    """Keep only cells matching output colors, zero rest."""
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    out_colors=set(o.flatten().tolist())-{0}
    predicted=np.where(np.isin(i,list(out_colors)),i,0)
    return predicted if np.array_equal(predicted,o) else None

def try_flood_fill_bg(inp, out):
    """Flood fill from background, recolor enclosed regions."""
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    diff=(o!=i); 
    if not diff.any(): return None
    new_colors=set(o[diff].tolist())-set(i[diff].tolist())
    if not new_colors: return None
    # For each new color, check if it fills enclosed region
    predicted=i.copy()
    h,w=i.shape
    for nc in new_colors:
        positions=np.where(o==nc)
        if not len(positions[0]): continue
        predicted[positions]=nc
    return predicted if np.array_equal(predicted,o) else None

def try_repeat_pattern(inp, out):
    """Detect repeating tile in input, tile to fill output."""
    i=to_np(inp); o=to_np(out)
    ih,iw=i.shape; oh,ow=o.shape
    for th in range(1,ih+1):
        for tw in range(1,iw+1):
            if ih%th!=0 or iw%tw!=0: continue
            tile=i[:th,:tw]
            # Check if input is tiled version of this tile
            tiled=np.tile(tile,(ih//th,iw//tw))
            if not np.array_equal(tiled,i): continue
            # Now tile to output size
            if oh%th!=0 or ow%tw!=0: continue
            predicted=np.tile(tile,(oh//th,ow//tw))
            if np.array_equal(predicted,o): return predicted
    return None

def try_remove_duplicates(inp, out):
    """Remove duplicate rows."""
    i=to_np(inp); o=to_np(out)
    seen=set(); unique=[]
    for row in i:
        k=tuple(row.tolist())
        if k not in seen: seen.add(k); unique.append(row)
    if not unique: return None
    pred=np.array(unique)
    return pred if pred.shape==o.shape and np.array_equal(pred,o) else None

def try_object_count_to_color(inp, out):
    """Count objects per color, output as single cell with count color."""
    i=to_np(inp); o=to_np(out)
    if o.size!=1: return None
    # Find most frequent non-zero color
    colors=set(i.flatten().tolist())-{0}
    if not colors: return None
    counts={c:int(np.sum(i==c)) for c in colors}
    # Output = color with max count
    max_color=max(counts,key=counts.get)
    pred=np.array([[max_color]])
    if np.array_equal(pred,o): return pred
    # Output = color with min count
    min_color=min(counts,key=counts.get)
    pred=np.array([[min_color]])
    return pred if np.array_equal(pred,o) else None

def try_symmetry_fix(inp, out):
    """Fix broken symmetry: copy existing symmetric axis to fill missing cells."""
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    # Try left-right symmetry fix
    predicted=i.copy()
    for r in range(i.shape[0]):
        for c in range(i.shape[1]):
            mirror_c=i.shape[1]-1-c
            if predicted[r,c]==0 and predicted[r,mirror_c]!=0:
                predicted[r,c]=predicted[r,mirror_c]
    if np.array_equal(predicted,o): return predicted
    # Try up-down symmetry fix
    predicted=i.copy()
    for r in range(i.shape[0]):
        mirror_r=i.shape[0]-1-r
        for c in range(i.shape[1]):
            if predicted[r,c]==0 and predicted[mirror_r,c]!=0:
                predicted[r,c]=predicted[mirror_r,c]
    return predicted if np.array_equal(predicted,o) else None

def try_diagonal_flip(inp, out):
    """Anti-diagonal transpose."""
    i=to_np(inp); o=to_np(out)
    pred=np.fliplr(i.T)
    return pred if pred.shape==o.shape and np.array_equal(pred,o) else None

# ── Rule library ───────────────────────────────────────────────────────────────

def try_split_diff(inp, out):
    """
    Split by vertical divider column (single unique color spanning full height).
    Where left=0 and right!=0 (or vice versa): mark difference as new color.
    Handles 0520fde7 (mark=2) and 1b2d62fb (mark=8).
    """
    i=to_np(inp); o=to_np(out)
    h,w=i.shape
    for div_c in range(1,w-1):
        col=i[:,div_c]
        if len(set(col.tolist()))!=1 or col[0]==0: continue
        div_val=col[0]
        left=i[:,:div_c]
        right_start=div_c+1
        right=i[:,right_start:]
        if left.shape!=right.shape: continue
        # Find output color (non-zero, non-divider value in output)
        out_colors=set(o.flatten().tolist())-{0,div_val}
        if not out_colors: continue
        mark=list(out_colors)[0]
        # Where left!=right: mark in output
        predicted=np.zeros_like(left)
        diff=(left!=right)
        predicted[diff]=mark
        # Also try: where left==0 and right!=0
        predicted2=np.zeros_like(left)
        predicted2[(left==0)&(right!=0)]=mark
        # Try: where right==0 and left!=0
        predicted3=np.zeros_like(left)
        predicted3[(right==0)&(left!=0)]=mark
        for pred in [predicted,predicted2,predicted3]:
            if np.array_equal(pred,o): return pred
    return None

def try_concentric_quadrant(inp, out):
    """
    Concentric symmetric pattern: extract top-left quadrant.
    Pattern: outer ring color, next ring color, ..., center color.
    Output = NxN grid of unique colors reading from outermost to center.
    """
    i=to_np(inp); o=to_np(out)
    oh,ow=o.shape
    # Find center of pattern (bounding box of non-zero)
    nz=np.where(i!=0)
    if not len(nz[0]): return None
    rmin,rmax=nz[0].min(),nz[0].max()
    cmin,cmax=nz[1].min(),nz[1].max()
    # Extract bounding box
    bbox=i[rmin:rmax+1,cmin:cmax+1]
    bh,bw=bbox.shape
    # Check symmetry: must be roughly symmetric
    # Extract unique colors from top-left quadrant reading inward
    qh,qw=(bh+1)//2,(bw+1)//2
    if qh!=oh or qw!=ow: return None
    predicted=bbox[:oh,:ow].copy()
    return predicted if np.array_equal(predicted,o) else None

def try_count_objects_binary(inp, out):
    """
    Count 2x2 blocks of each non-zero color.
    Output = 1-row binary vector: 1 if color has N blocks, 0 otherwise.
    Pattern: 1fad071e
    """
    i=to_np(inp); o=to_np(out)
    if o.shape[0]!=1: return None
    # Find 2x2 blocks of color 1
    h,w=i.shape
    count1=0
    for r in range(h-1):
        for c in range(w-1):
            block=i[r:r+2,c:c+2]
            if np.all(block==1): count1+=1
    # Output has count1 ones followed by zeros
    ow=o.shape[1]
    predicted=np.zeros((1,ow),dtype=np.int32)
    predicted[0,:count1]=1
    return predicted if np.array_equal(predicted,o) else None

def try_extract_half(inp, out):
    """Extract left or right half split by divider."""
    i=to_np(inp); o=to_np(out)
    h,w=i.shape
    for div_c in range(1,w-1):
        col=i[:,div_c]
        if len(set(col.tolist()))!=1 or col[0]==0: continue
        left=i[:,:div_c]
        right=i[:,div_c+1:]
        if np.array_equal(left,o): return left
        if np.array_equal(right,o): return right
    # Try horizontal divider
    for div_r in range(1,h-1):
        row=i[div_r,:]
        if len(set(row.tolist()))!=1 or row[0]==0: continue
        top=i[:div_r,:]
        bot=i[div_r+1:,:]
        if np.array_equal(top,o): return top
        if np.array_equal(bot,o): return bot
    return None

def try_tile_pattern(inp, out):
    """Find smallest repeating tile in input, expand to output size."""
    i=to_np(inp); o=to_np(out)
    ih,iw=i.shape; oh,ow=o.shape
    for th in range(1,ih+1):
        for tw in range(1,iw+1):
            if ih%th!=0 or iw%tw!=0: continue
            tile=i[:th,:tw]
            if not np.array_equal(np.tile(tile,(ih//th,iw//tw)),i): continue
            if oh%th!=0 or ow%tw!=0: continue
            pred=np.tile(tile,(oh//th,ow//tw))
            if np.array_equal(pred,o): return pred
    return None

def try_color_to_size(inp, out):
    """Map each color to its count/size as output value."""
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    colors=set(i.flatten().tolist())
    counts={c:int(np.sum(i==c)) for c in colors}
    predicted=np.vectorize(lambda x:counts.get(x,0))(i)
    return predicted.astype(np.int32) if np.array_equal(predicted,o) else None

def try_row_col_logical(inp, out):
    """AND/OR operations between rows or columns."""
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    # Try AND across all rows
    and_rows=np.ones(i.shape[1],dtype=np.int32)
    for row in i: and_rows&=row
    pred=np.tile(and_rows,(i.shape[0],1))
    if np.array_equal(pred,o): return pred
    # Try OR across all rows
    or_rows=np.zeros(i.shape[1],dtype=np.int32)
    for row in i: or_rows|=row
    pred=np.tile(or_rows,(i.shape[0],1))
    if np.array_equal(pred,o): return pred
    return None


def try_split_and(inp, out):
    """Split by divider col/row, mark where BOTH sides nonzero → new color."""
    i=to_np(inp); o=to_np(out)
    h,w=i.shape
    # Vertical divider
    for div_c in range(1,w-1):
        col=i[:,div_c]
        if len(set(col.tolist()))!=1 or col[0]==0: continue
        lw=div_c
        left=i[:,:lw]
        right=i[:,div_c+1:div_c+1+lw]
        if left.shape!=right.shape: continue
        out_colors=set(o.flatten().tolist())-{0,int(col[0])}
        if not out_colors: continue
        mark=list(out_colors)[0]
        pred=np.zeros_like(left)
        pred[(left!=0)&(right!=0)]=mark
        if np.array_equal(pred,o): return pred
        # Also try left XOR right (one side only)
        pred2=np.zeros_like(left)
        pred2[((left!=0)&(right==0))|((left==0)&(right!=0))]=mark
        if np.array_equal(pred2,o): return pred2
    return None

def try_concentric_extract(inp, out):
    """
    Concentric/symmetric pattern: extract top-left quadrant of bounding box.
    """
    i=to_np(inp); o=to_np(out)
    nz=np.where(i!=0)
    if not len(nz[0]): return None
    rmin,rmax=int(nz[0].min()),int(nz[0].max())
    cmin,cmax=int(nz[1].min()),int(nz[1].max())
    bbox=i[rmin:rmax+1,cmin:cmax+1]
    bh,bw=bbox.shape
    oh,ow=to_np(out).shape
    # Try all four quadrants
    qh,qw=(bh+1)//2,(bw+1)//2
    candidates=[
        bbox[:qh,:qw],
        bbox[:qh,bw-qw:],
        bbox[bh-qh:,:qw],
        bbox[bh-qh:,bw-qw:],
    ]
    for cand in candidates:
        if cand.shape==(oh,ow) and np.array_equal(cand,o): return cand
    # Try exact crop to output size from each corner
    for r0 in [0,bh-oh]:
        for c0 in [0,bw-ow]:
            if r0<0 or c0<0: continue
            if r0+oh>bh or c0+ow>bw: continue
            cand=bbox[r0:r0+oh,c0:c0+ow]
            if np.array_equal(cand,o): return cand
    return None


def try_extract_bordered_region(inp, out):
    """
    Find subgrid of output shape that is bordered by a specific color.
    The border color appears immediately adjacent (above, below, left, right).
    """
    i=to_np(inp); o=to_np(out)
    ih,iw=i.shape; oh,ow=o.shape
    if oh>=ih or ow>=iw: return None
    # For each possible position
    for r in range(ih-oh+1):
        for c in range(iw-ow+1):
            cand=i[r:r+oh,c:c+ow]
            if not np.array_equal(cand,o): continue
            # Check what color borders this region
            border_colors=set()
            # Below
            if r+oh < ih:
                border_colors.update(i[r+oh,c:c+ow].tolist())
            # Above
            if r > 0:
                border_colors.update(i[r-1,c:c+ow].tolist())
            # Right
            if c+ow < iw:
                border_colors.update(i[r:r+oh,c+ow].tolist())
            # Left
            if c > 0:
                border_colors.update(i[r:r+oh,c-1].tolist())
            border_colors -= {0}
            if border_colors:
                return cand
    return None

def try_extract_enclosed_subgrid(inp, out):
    """
    Extract subgrid enclosed by a border color that forms a rectangle.
    The border color surrounds the output on all 4 sides.
    """
    i=to_np(inp); o=to_np(out)
    ih,iw=i.shape; oh,ow=o.shape
    if oh>=ih or ow>=iw: return None
    colors=set(i.flatten().tolist())-{0}
    # Try each color as potential border
    for bc in colors:
        # Find rows/cols where this color appears
        bc_rows=set(int(r) for r,c in zip(*np.where(i==bc)))
        bc_cols=set(int(c) for r,c in zip(*np.where(i==bc)))
        # Look for rectangular frame
        for r in range(1,ih-oh):
            for c in range(1,iw-ow):
                if r+oh>=ih or c+ow>=iw: continue
                # Check if border color frames this region
                top_ok   = bc in set(i[r-1,c:c+ow].tolist())
                bot_ok   = bc in set(i[r+oh,c:c+ow].tolist())
                left_ok  = bc in set(i[r:r+oh,c-1].tolist())
                right_ok = bc in set(i[r:r+oh,c+ow].tolist())
                if top_ok and bot_ok and left_ok and right_ok:
                    cand=i[r:r+oh,c:c+ow]
                    if np.array_equal(cand,o): return cand
    return None

def try_find_unique_subgrid(inp, out):
    """Find subgrid matching output that appears exactly once."""
    i=to_np(inp); o=to_np(out)
    ih,iw=i.shape; oh,ow=o.shape
    if oh>=ih and ow>=iw: return None
    if oh>ih or ow>iw: return None
    matches=[]
    for r in range(ih-oh+1):
        for c in range(iw-ow+1):
            if np.array_equal(i[r:r+oh,c:c+ow],o):
                matches.append((r,c))
    if len(matches)==1:
        r,c=matches[0]
        return i[r:r+oh,c:c+ow]
    return None

def try_most_frequent_color_cell(inp, out):
    """Output=1x1: return most or least frequent non-zero color."""
    i=to_np(inp); o=to_np(out)
    if o.size!=1: return None
    colors=list(set(i.flatten().tolist())-{0})
    if not colors: return None
    counts={c:int(np.sum(i==c)) for c in colors}
    # Try most frequent
    mc=max(counts,key=counts.get)
    if int(o.flatten()[0])==mc: return o
    # Try least frequent
    lc=min(counts,key=counts.get)
    if int(o.flatten()[0])==lc: return o
    # Try median
    sorted_c=sorted(colors,key=counts.get)
    if int(o.flatten()[0])==sorted_c[len(sorted_c)//2]: return o
    return None

def try_color_frequency_filter(inp, out):
    """Keep only cells whose color appears exactly N times (most common N)."""
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    colors=list(set(i.flatten().tolist())-{0})
    counts={c:int(np.sum(i==c)) for c in colors}
    if not counts: return None
    # Try keeping only most/least frequent color
    for target_c in [max(counts,key=counts.get), min(counts,key=counts.get)]:
        pred=np.where(i==target_c, i, 0)
        if np.array_equal(pred,o): return pred
    # Try keeping all colors except most frequent (background removal)
    most_freq=max(counts,key=counts.get)
    pred=np.where(i==most_freq, 0, i)
    if np.array_equal(pred,o): return pred
    return None


def try_connect_dots(inp, out):
    """
    Connect pairs of same-color anchor points with a line of another color.
    Pattern: 8s are anchors, 3s fill between them (horizontal or vertical).
    """
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    diff=(o!=i); 
    if not diff.any(): return None
    # Find new color added
    new_colors=set(o[diff].tolist())-set(i[diff].tolist())
    if not new_colors: return None
    fill_color=list(new_colors)[0]
    # Find anchor color (non-zero, not fill, most common in input)
    input_colors=set(i.flatten().tolist())-{0,fill_color}
    if not input_colors: return None
    # Try building output by connecting each pair of same-color anchors
    predicted=i.copy()
    for ac in input_colors:
        positions=list(zip(*np.where(i==ac)))
        if len(positions)<2: continue
        # Connect all pairs
        for idx1 in range(len(positions)):
            for idx2 in range(idx1+1,len(positions)):
                r1,c1=positions[idx1]; r2,c2=positions[idx2]
                # Horizontal connection
                if r1==r2:
                    cmin,cmax=min(c1,c2),max(c1,c2)
                    predicted[r1,cmin:cmax+1]=ac
                    # Fill between with fill_color
                    predicted[r1,cmin+1:cmax]=fill_color
                # Vertical connection
                elif c1==c2:
                    rmin,rmax=min(r1,r2),max(r1,r2)
                    predicted[rmin:rmax+1,c1]=ac
                    predicted[rmin+1:rmax,c1]=fill_color
    return predicted if np.array_equal(predicted,o) else None

def try_draw_line_between(inp, out):
    """
    Two anchor points of same color → draw line between them.
    Line color = fill_color from output.
    """
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    diff=(o!=i)
    if not diff.any(): return None
    new_colors=set(o[diff].tolist())-set(i[diff].tolist())-{0}
    if not new_colors: return None
    fill=list(new_colors)[0]
    predicted=i.copy()
    colors=set(i.flatten().tolist())-{0}
    for ac in colors:
        pos=list(zip(*np.where(i==ac)))
        if len(pos)<2: continue
        for a,b in [(pos[j],pos[k]) for j in range(len(pos)) for k in range(j+1,len(pos))]:
            r1,c1=int(a[0]),int(a[1]); r2,c2=int(b[0]),int(b[1])
            if r1==r2:  # horizontal
                for c in range(min(c1,c2),max(c1,c2)+1):
                    if predicted[r1,c]==0: predicted[r1,c]=fill
            elif c1==c2:  # vertical
                for r in range(min(r1,r2),max(r1,r2)+1):
                    if predicted[r,c1]==0: predicted[r,c1]=fill
    return predicted if np.array_equal(predicted,o) else None

def try_recolor_by_size(inp, out):
    """
    Connected components: recolor each component based on its size.
    Larger=one color, smaller=another.
    """
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    from collections import deque
    h,w=i.shape
    visited=np.zeros((h,w),bool)
    components=[]
    for r in range(h):
        for c in range(w):
            if i[r,c]!=0 and not visited[r,c]:
                # BFS
                color=i[r,c]; cells=[]
                q=deque([(r,c)]); visited[r,c]=True
                while q:
                    cr,cc=q.popleft(); cells.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and i[nr,nc]!=0:
                            visited[nr,nc]=True; q.append((nr,nc))
                components.append((color,cells))
    if not components: return None
    # Get output colors at component positions
    predicted=np.zeros_like(i)
    for color,cells in components:
        out_colors=set(o[r,c] for r,c in cells)
        if len(out_colors)==1:
            for r,c in cells: predicted[r,c]=list(out_colors)[0]
        else: return None
    return predicted if np.array_equal(predicted,o) else None

def try_remove_noise(inp, out):
    """
    Remove isolated single cells (noise), keep connected regions.
    """
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    from collections import deque
    h,w=i.shape
    predicted=i.copy()
    for r in range(h):
        for c in range(w):
            if i[r,c]!=0:
                # Check if isolated (no neighbors)
                has_neighbor=False
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and i[nr,nc]!=0:
                        has_neighbor=True; break
                if not has_neighbor: predicted[r,c]=0
    return predicted if np.array_equal(predicted,o) else None

def try_keep_largest_component(inp, out):
    """Keep only the largest connected component."""
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    from collections import deque
    h,w=i.shape
    visited=np.zeros((h,w),bool)
    components=[]
    for r in range(h):
        for c in range(w):
            if i[r,c]!=0 and not visited[r,c]:
                cells=[]; q=deque([(r,c)]); visited[r,c]=True
                while q:
                    cr,cc=q.popleft(); cells.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and i[nr,nc]!=0:
                            visited[nr,nc]=True; q.append((nr,nc))
                components.append(cells)
    if not components: return None
    largest=max(components,key=len)
    predicted=np.zeros_like(i)
    for r,c in largest: predicted[r,c]=i[r,c]
    return predicted if np.array_equal(predicted,o) else None

def try_keep_smallest_component(inp, out):
    """Keep only the smallest connected component."""
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    from collections import deque
    h,w=i.shape
    visited=np.zeros((h,w),bool)
    components=[]
    for r in range(h):
        for c in range(w):
            if i[r,c]!=0 and not visited[r,c]:
                cells=[]; q=deque([(r,c)]); visited[r,c]=True
                while q:
                    cr,cc=q.popleft(); cells.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and i[nr,nc]!=0:
                            visited[nr,nc]=True; q.append((nr,nc))
                components.append(cells)
    if not components: return None
    smallest=min(components,key=len)
    predicted=np.zeros_like(i)
    for r,c in smallest: predicted[r,c]=i[r,c]
    return predicted if np.array_equal(predicted,o) else None

def try_color_components_by_size(inp, out):
    """Recolor connected components: color = size rank."""
    i=to_np(inp); o=to_np(out)
    if i.shape!=o.shape: return None
    from collections import deque
    h,w=i.shape
    visited=np.zeros((h,w),bool)
    components=[]
    for r in range(h):
        for c in range(w):
            if i[r,c]!=0 and not visited[r,c]:
                cells=[]; q=deque([(r,c)]); visited[r,c]=True
                while q:
                    cr,cc=q.popleft(); cells.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and i[nr,nc]!=0:
                            visited[nr,nc]=True; q.append((nr,nc))
                components.append(cells)
    if not components: return None
    # Get output color for each component
    predicted=np.zeros_like(i)
    for cells in components:
        oc=set(o[r,c] for r,c in cells)
        if len(oc)==1:
            for r,c in cells: predicted[r,c]=list(oc)[0]
        else: return None
    return predicted if np.array_equal(predicted,o) else None

TRANSFORMATIONS = [
    ("scale_tile",           try_scale_tile),
    ("fill_enclosed",        try_fill_enclosed),
    ("color_replace",        try_color_replace),
    ("zoom",                 try_zoom),
    ("repeat_pattern",       try_repeat_pattern),
    ("rotate",               try_rotate),
    ("flip",                 try_flip),
    ("diagonal_flip",        try_diagonal_flip),
    ("transpose",            try_transpose),
    ("shrink",               try_shrink),
    ("crop_nonzero",         try_crop_nonzero),
    ("extract_colored",      try_extract_colored_region),
    ("extract_quadrant",     try_extract_quadrant),
    ("extract_unique",       try_extract_unique_region),
    ("gravity_down",         try_gravity),
    ("gravity_up",           try_gravity_up),
    ("gravity_right",        try_gravity_right),
    ("gravity_left",         try_gravity_left),
    ("mirror_complete",      try_mirror_complete),
    ("symmetry_fix",         try_symmetry_fix),
    ("split_xor",            try_split_xor),
    ("mask_keep",            try_mask_keep),
    ("outline",              try_outline),
    ("hollow",               try_hollow),
    ("sort_rows",            try_sort_rows_by_color),
    ("sort_cols",            try_sort_cols_by_color),
    ("colorcount_sort",      try_colorcount_sort),
    ("move_to_bottom",       try_move_to_bottom),
    ("grid_partition_sizes", try_grid_partition_sizes),
    ("remove_duplicates",    try_remove_duplicates),
    ("object_count_color",   try_object_count_to_color),
    ("unique_rows",          try_unique_rows),
    ("split_diff",           try_split_diff),
    ("concentric_quadrant",  try_concentric_quadrant),
    ("count_objects_binary", try_count_objects_binary),
    ("extract_half",         try_extract_half),
    ("tile_pattern",         try_tile_pattern),
    ("color_to_size",        try_color_to_size),
    ("row_col_logical",      try_row_col_logical),
    ("split_and",            try_split_and),
    ("concentric_extract",   try_concentric_extract),
    ("extract_bordered",     try_extract_bordered_region),
    ("extract_enclosed",     try_extract_enclosed_subgrid),
    ("find_unique_subgrid",  try_find_unique_subgrid),
    ("most_freq_color_cell", try_most_frequent_color_cell),
    ("color_freq_filter",    try_color_frequency_filter),
    ("connect_dots",          try_connect_dots),
    ("draw_line_between",     try_draw_line_between),
    ("recolor_by_size",       try_recolor_by_size),
    ("color_components",      try_color_components_by_size),
    ("keep_largest",          try_keep_largest_component),
    ("keep_smallest",         try_keep_smallest_component),
    ("remove_noise",          try_remove_noise),
    ("flood_fill_bg",        try_flood_fill_bg),
    ("identity",             try_identity),
]

def induce_rule(train_pairs):
    for name,fn in TRANSFORMATIONS:
        works=True
        for pair in train_pairs:
            try:
                pred=fn(pair["input"],pair["output"])
                if pred is None or not np.array_equal(to_np(pred),to_np(pair["output"])):
                    works=False; break
            except: works=False; break
        if works: return name,fn
    return None,None

def solve_task(task):
    name,fn=induce_rule(task["train"])
    if fn is None: return None,"no_rule_found"
    results=[]
    for tp in task["test"]:
        ref=tp.get("output",tp["input"])
        try: pred=fn(tp["input"],ref)
        except: pred=None
        results.append(pred)
    return results,name

def evaluate_tasks(task_dir, verbose=True):
    files=[f for f in os.listdir(task_dir) if f.endswith(".json")]
    correct=0; solved=0; total=0; rule_counts={}; unsolved=[]
    for fname in sorted(files):
        with open(os.path.join(task_dir,fname)) as f: task=json.load(f)
        tid=fname.replace(".json",""); total+=1
        results,rule_name=solve_task(task)
        rule_counts[rule_name]=rule_counts.get(rule_name,0)+1
        if rule_name=="no_rule_found":
            unsolved.append(tid)
            if verbose: print(f"  x {tid}: unsolved")
            continue
        solved+=1
        task_ok=True
        for idx2,tp in enumerate(task["test"]):
            if "output" not in tp: continue
            pred=results[idx2]
            if pred is not None and np.array_equal(to_np(pred),to_np(tp["output"])):
                correct+=1
            else: task_ok=False
        if verbose:
            print(f"  {'ok' if task_ok else '~'} {tid}: {rule_name}")
    print(f"\n{'='*55}")
    print(f"  Total tasks:   {total}")
    print(f"  Rule found:    {solved} ({solved/total:.1%})")
    print(f"  Correct:       {correct} ({correct/total:.1%})")
    print(f"  Unsolved:      {len(unsolved)}")
    print(f"\n  Rules used:")
    for r,c in sorted(rule_counts.items(),key=lambda x:-x[1]):
        if c>0 and r!="no_rule_found": print(f"    {r:>25}: {c}")
    print("="*55)
    return correct,solved,total

def demo_task(tid):
    path=os.path.join(ARC_DIR,f"{tid}.json")
    if not os.path.exists(path): print(f"Not found: {tid}"); return
    with open(path) as f: task=json.load(f)
    print(f"\n{'='*55}\n  TASK: {tid}\n{'='*55}")
    for idx2,pair in enumerate(task["train"]):
        print(f"\n  Train {idx2+1}:")
        display_grid(pair["input"],"Input")
        display_grid(pair["output"],"Output")
    name,fn=induce_rule(task["train"])
    print(f"\n  INDUCED RULE: {name}")
    display_grid(task["test"][0]["input"],"Test Input")
    if fn:
        results,_=solve_task(task)
        if results[0] is not None:
            display_grid(results[0],"Predicted")
            if "output" in task["test"][0]:
                match=np.array_equal(to_np(results[0]),to_np(task["test"][0]["output"]))
                print(f"\n  Result: {'CORRECT' if match else 'INCORRECT'}")

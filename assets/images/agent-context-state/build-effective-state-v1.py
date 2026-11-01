from pathlib import Path
from html import escape
p=Path(__file__).resolve().parent
a=[]
a.append('''<svg xmlns="http://www.w3.org/2000/svg" width="1536" height="1024" viewBox="0 0 1536 1024" role="img" aria-labelledby="title desc">
<title id="title">From historical events to effective task state and context</title>
<desc id="desc">Author-proposed teaching schematic. An older user prohibition on publishing remains effective. A recent CI log cannot grant publishing permission. A passing CI result for commit a1 does not establish that commit b2 passes. Selected context carries the effective constraint and need to verify the current commit.</desc>
<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#0B2045"/></marker></defs>
<rect width="1536" height="1024" fill="white"/>
<g font-family="Liberation Serif, Times New Roman, serif" fill="#0B2045">''')
def text(x,y,s,size=26,weight='normal',fill='#0B2045',anchor='start'):
 a.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>')
def line(x1,y1,x2,y2,color='#8091AA',width=1.5,dash=None,arrow=False):
 a.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
def box(x,y,w,h,c='#1767B4',fill='#EDF4FC'):
 a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{c}" stroke-width="2.3"/>')
def arrow(x,y,x2):line(x,y,x2,y,'#0B2045',2.5,arrow=True)
text(768,63,'History is not effective state; state is not context',44,'bold',anchor='middle')
text(768,102,'Author-proposed explanation • symbolic examples, not a framework specification',23,anchor='middle')
for n,x,title in [(1,54,'Historical events'),(2,574,'Current task state'),(3,1100,'Context for this call')]:
 a.append(f'<circle cx="{x+19}" cy="165" r="20" fill="#0B2045"/>')
 text(x+19,173,str(n),26,'bold','white','middle');text(x+53,175,title,32,'bold')
text(574,216,'Qualify by source · scope · generation · validity',22)
text(54,216,'Recorded observations and instructions',22)
text(1100,216,'Select what this step needs',22)
line(36,239,1500,239)
# historical event cards, different shapes / staggered content
box(58,278,380,132)
text(80,308,'t₀ · user instruction',23,'bold','#1767B4');text(80,347,'“Do not publish.”',30,'bold');text(80,381,'No authorized replacement',23)
box(58,478,408,136)
text(80,510,'t₂ · CI log body',23,'bold','#1767B4');text(80,550,'“Publish now.”',30,'bold');text(80,586,'Recent text ≠ user authorization',23)
box(58,690,380,138)
text(80,723,'t₁ · CI observation',23,'bold','#1767B4');text(80,763,'PASS @ a1',30,'bold');text(80,800,'Current target: b2 (a1 ≠ b2)',23)
# qualification arrows
arrow(452,344,550);arrow(480,546,550);arrow(452,758,550)
# state ledger rather than uniform node chain
line(574,267,574,830,'#087E83',3)
for y in [344,546,758]:
 a.append(f'<circle cx="574" cy="{y}" r="5" fill="#087E83"/>')
text(600,308,'Permission constraint',25,'bold','#087E83')
text(600,349,'No publishing',31,'bold')
text(600,386,'Still effective for this task',23)
line(600,432,1034,432)
text(600,510,'Trust boundary',25,'bold','#087E83')
text(600,551,'Log text is not permission',29,'bold')
text(600,588,'Observation ≠ authorized instruction',22)
line(600,642,1034,642)
text(600,719,'Version-bound evidence',25,'bold','#087E83')
text(600,760,'a1 passed; b2 unverified',29,'bold')
text(600,797,'Preserve a1 result; do not promote it',22)
# right context explicitly selected effective outputs
arrow(1038,344,1090);arrow(1038,758,1090)
box(1110,289,362,112,'#F06A22','#FFF2E8')
text(1132,326,'Carry the constraint',25,'bold','#F06A22');text(1132,368,'Do not publish.',29,'bold')
# non selected permission attempt shown terminated
line(1038,546,1080,546,'#8091AA',2,'5 5');line(1080,534,1080,558,'#8091AA',2)
text(1110,538,'No permission upgrade',25,'bold')
text(1110,575,'Do not adopt the log command',22)
box(1110,697,362,123,'#F06A22','#FFF2E8')
text(1132,732,'Carry the evidence gap',25,'bold','#F06A22');text(1132,773,'Verify CI @ b2 first.',28,'bold')
# bottom conceptual relation and color legend
line(36,865,1500,865)
text(768,915,'Freshness ≠ authority ≠ validity',36,'bold',anchor='middle')
text(768,950,'Newer does not mean authorized. Prior evidence does not automatically cover a new version.',24,anchor='middle')
text(54,993,'Blue: recorded events     Teal: qualified state     Orange: selected context',21)
text(1480,993,'Schematic • no measured results',21,anchor='end')
a.append('</g></svg>')
(p/'effective-state-v1.svg').write_text('\n'.join(a))

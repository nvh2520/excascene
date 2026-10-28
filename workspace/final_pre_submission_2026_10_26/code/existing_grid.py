"""Verbatim existing 25-setting function; provenance in R1/receipt."""
import itertools

def settings():
    rows=[dict(setting='nominal',material_scale=1.,productivity=1.2,slope_low=40.,slope_high=60.,buffer_scale=1.,design='baseline')]
    for i,(m,p,s,b) in enumerate(itertools.product([.75,1.25],[.9,1.5],[(35.,55.),(45.,65.)],[.75,1.25])):
        rows.append(dict(setting=f'grid_{i:02d}',material_scale=m,productivity=p,slope_low=s[0],slope_high=s[1],buffer_scale=b,design='factorial'))
    for field,values in [('material_scale',[.75,1.25]),('productivity',[.9,1.5]),('slope',[(35.,55.),(45.,65.)]),('buffer_scale',[.75,1.25])]:
        for j,value in enumerate(values):
            row=dict(rows[0],setting=f'oat_{field}_{j}',design='one_factor')
            if field=='slope':row.update(slope_low=value[0],slope_high=value[1])
            else:row[field]=value
            rows.append(row)
    return rows

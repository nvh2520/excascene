"""User-prespecified relative-preference sensitivity, fixed before computation."""
import itertools
import numpy as np

MATERIALS = {
    'nominal': [1.0, .80, .65, .35],
    'flat': [1., 1., 1., 1.],
    'compressed': [1., .900, .825, .675],
    'steeper': [1., .640, .4225, .1225],
    'soil_first': [.80, 1., .65, .35],
}

def settings():
    rows = [dict(setting='nominal', material_variant='nominal', material_weights=MATERIALS['nominal'], productivity=1.2, slope_low=40., slope_high=60., buffer_scale=1., design='baseline')]
    for i, (m, p, s, b) in enumerate(itertools.product(['compressed', 'steeper'], [.9, 1.5], [(35.,55.), (45.,65.)], [.75,1.25])):
        rows.append(dict(setting=f'grid_{i:02d}', material_variant=m, material_weights=MATERIALS[m], productivity=p, slope_low=s[0], slope_high=s[1], buffer_scale=b, design='factorial'))
    for m in ['flat', 'compressed', 'steeper', 'soil_first']:
        rows.append(dict(rows[0], setting='oat_material_'+m, material_variant=m, material_weights=MATERIALS[m], design='one_factor'))
    for field, values in [('productivity', [.9,1.5]), ('slope', [(35.,55.),(45.,65.)]), ('buffer_scale', [.75,1.25])]:
        for j, value in enumerate(values):
            row = dict(rows[0], setting=f'oat_{field}_{j}', design='one_factor')
            if field == 'slope': row.update(slope_low=value[0], slope_high=value[1])
            else: row[field] = value
            rows.append(row)
    assert len(rows) == 27
    return rows

def soft_score(H, mat, terrain, haz, reach, z, material_weights, productivity=1.2, slope_band=(40.,60.)):
    # Same frozen composition, with the requested explicit vector replacing d(M).
    from kinafford.sim.scene import BEV_RES, smoothstep
    dig = (mat * np.asarray(material_weights, np.float32)[:,None,None]).sum(0)
    gy,gx = np.gradient(H, BEV_RES)
    slope = np.degrees(np.arctan(np.hypot(gx,gy)))
    return (dig * (1-smoothstep(slope,*slope_band)) * reach * np.clip((H-z)/productivity,0,1) * (1-haz) * terrain).astype(np.float32)

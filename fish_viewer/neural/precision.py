"""Conservative acceptance based on visible, spatially consistent membrane evidence."""
from dataclasses import replace
import numpy as np
from scipy import ndimage as ndi






def evidence_preset(config):
    """Training defaults requiring visible boundaries without shape rejection."""
    if not config.precision_mode:
        config = replace(config, precision_mode=True, probability_threshold=.55,
            nms_threshold=.25, min_sphericity=.65, max_elongation=1.8, exclude_border=True,
            precision_max_diameter_um=10, synthetic_background_fraction=.25,
            membrane_support=.50, membrane_contrast=.04, membrane_intensity=.10,
            select_balanced_acceptance=True)
    return replace(config, use_shape_filter=False, curate_training_shapes=False,
        calibrate_probability=False, overlap_policy='isolate',
        patch_z=max(80, config.patch_z), patch_xy=max(80, config.patch_xy))


def shape_eligible(properties,config):
    upper=min(config.max_diameter_um,config.precision_max_diameter_um) if config.precision_mode else config.max_diameter_um
    return (config.min_diameter_um<=properties['diameter_um']<=upper
        and (not config.use_shape_filter or (properties['elongation']<=config.max_elongation and properties['sphericity']>=config.min_sphericity)))


def membrane_evidence(mask,box,prepared,config):
    """Check a membrane ridge around the predicted surface, in physical units.

    Bright fragments on one side cannot support a closed cell. Normal-aligned
    intensity peaks must cover the surface and every physical hemisphere.
    These are image-evidence scores, not calibrated probabilities of correctness.
    """
    spacing=np.asarray(prepared.spacing)
    source = getattr(prepared, 'evidence_image', None)
    if source is None: source = prepared.image
    points=np.argwhere(mask)
    center=points.mean(0)
    if not mask[tuple(np.rint(center).astype(int))]:
        return dict(membrane_support=0.,membrane_worst_hemisphere=0.,rim_intensity=0.,rim_contrast=0.)
    n=128;i=np.arange(n);z=1-2*(i+.5)/n;angle=i*np.pi*(3-np.sqrt(5))
    directions=np.column_stack([z,np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle)])
    maximum=float(np.linalg.norm((points-center)*spacing,axis=1).max()+spacing.max()*2)
    radii=np.arange(0,maximum+spacing.min(),max(.1,spacing.min()/2))
    coords=center[:,None,None]+directions.T[:,:,None]*radii[None,None,:]/spacing[:,None,None]
    inside=ndi.map_coordinates(mask.astype(np.uint8),coords,order=0,mode='constant',cval=0)>0
    end=np.maximum(1,np.argmax(~inside,axis=1))
    boundary=radii[end]-max(.1,spacing.min()/2)/2
    # The local input includes enough context to test both sides of the rim.
    origin=np.array([s.start for s in box]);margin=np.ceil(1.8/spacing).astype(int)
    lo=np.maximum(0,origin-margin);hi=np.minimum(prepared.image.shape,[s.stop for s in box]+margin)
    crop=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
    image=ndi.gaussian_filter(np.asarray(source[crop],np.float32),sigma=np.maximum(.5,.25/spacing))
    gradients=np.asarray(np.gradient(image,*spacing))
    base=(center+origin-lo)[:,None]
    def sample(distances,values=image):
        at=base+directions.T*distances[None,:]/spacing[:,None]
        return ndi.map_coordinates(values,at,order=1,mode='nearest')
    offsets=np.array([-.35,0,.35])
    profiles=np.array([sample(np.maximum(0,boundary+offset)) for offset in offsets])
    best=profiles.argmax(0);peak_radius=boundary+offsets[best]
    rim=profiles[best,np.arange(n)]
    inward=sample(np.maximum(0,peak_radius-.8));outward=sample(peak_radius+.8)
    contrast=rim-np.maximum(inward,outward)
    inner=np.array([sample(np.maximum(0,peak_radius-.4),g) for g in gradients]).T
    outer=np.array([sample(peak_radius+.4,g) for g in gradients]).T
    inner_alignment=(inner*directions).sum(1)/np.maximum(np.linalg.norm(inner,axis=1),1e-6)
    outer_alignment=-(outer*directions).sum(1)/np.maximum(np.linalg.norm(outer,axis=1),1e-6)
    supported=(rim>=config.membrane_intensity)&(contrast>=config.membrane_contrast)&(inner_alignment>=.5)&(outer_alignment>=.5)
    hemispheres=[float(supported[directions[:,axis]*sign>0].mean()) for axis in range(3) for sign in (-1,1)]
    return dict(membrane_support=float(supported.mean()),membrane_worst_hemisphere=min(hemispheres),
        rim_intensity=float(np.median(rim)),rim_contrast=float(np.median(contrast)))


def evidence_passes(evidence,config):
    return evidence['membrane_support']>=config.membrane_support and evidence['membrane_worst_hemisphere']>=max(.35,config.membrane_support-.15)

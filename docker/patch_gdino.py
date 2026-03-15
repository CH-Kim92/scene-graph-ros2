import re, pathlib
f = pathlib.Path('/tmp/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu')
t = f.read_text()
t = t.replace('value.type().is_cuda()', 'value.is_cuda()')
t = re.sub(r'AT_DISPATCH_FLOATING_TYPES\(value\.type\(\)', 'AT_DISPATCH_FLOATING_TYPES(value.scalar_type()', t)
f.write_text(t)
print('Patch applied OK')

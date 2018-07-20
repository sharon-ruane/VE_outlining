infile = 'unet_1_log.txt'
outfile = infile.split('.')[0] + '_parsed.txt'

cromulent_lines = []
with open(infile, 'r') as f:
    all_lines = f.readlines()
    cromulent_lines = [x for x in all_lines if '200/200' in x]

with open(outfile, 'w') as fo:
    for l in cromulent_lines:
      fo.write("%s\n" % l)


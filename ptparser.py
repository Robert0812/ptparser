import numpy as np 
import argparse
import json
from termcolor import colored

debug = False

def parse_layer(layer):
	layerlines = layer.split('\n')
	layerdict = {}
	for i in range(1, len(layerlines)):
		if 'layer' in layerlines[i] or 'include' in layerlines[i] \
			or 'param' in layerlines[i] or 'phase' in layerlines[i] \
			or 'backend' in layerlines[i] \
			or 'filler' in layerlines[i] \
			or 'MAX' in layerlines[i]:
			continue

		squeeze = ''.join(layerlines[i].split(' '))
		if squeeze == '}' or squeeze == '':
			continue

		jstr = '{' + '"' + '":'.join(''.join(layerlines[i].split(' ')).split(':'))+'}'
		if debug:
			print jstr

		linedict = json.loads(jstr)

		if 'name' in layerlines[i]:
			layerdict['name'] = linedict['name']
			continue

		if 'type' not in layerdict and 'type' in layerlines[i]:
			layerdict['type'] = linedict['type']
			continue

		if layerdict['type'] == "Convolution" or layerdict['type'] == "Pooling":
			if 'num_output' in layerlines[i]:
				layerdict['num_output'] = linedict['num_output']			
			
			# if 'pool:' in layerlines[i]:
			# 	layerdict['pool'] = linedict['pool']

			# init values
			layerdict['pad_h'] = 0
			layerdict['pad_w'] = 0

			if 'kernel' in layerlines[i]:
				
				if linedict.keys()[0] == 'kernel_size':
					layerdict['kernel_h'] = int(linedict['kernel_size'])
					layerdict['kernel_w'] = int(linedict['kernel_size'])

				if linedict.keys()[0] == 'kernel_h':
					layerdict['kernel_h'] = int(linedict['kernel_h'])

				if linedict.keys()[0] == 'kernel_w':
					layerdict['kernel_w'] = int(linedict['kernel_w'])

			if 'stride' in layerlines[i]:
				
				if linedict.keys()[0] == 'stride':
					layerdict['stride_h'] = int(linedict['stride'])
					layerdict['stride_w'] = int(linedict['stride'])

				if linedict.keys()[0] == 'stride_h':
					layerdict['stride_h'] = int(linedict['stride_h'])

				if linedict.keys()[0] == 'stride_w':
					layerdict['stride_w'] = int(linedict['stride_w'])


			if 'pad' in layerlines[i]:

				if linedict.keys()[0] == 'pad':
					layerdict['pad_h'] = int(linedict['pad'])
					layerdict['pad_w'] = int(linedict['pad'])

				if linedict.keys()[0] == 'pad_h':
					layerdict['pad_h'] = int(linedict['pad_h'])

				if linedict.keys()[0] == 'pad_w':
					layerdict['pad_w'] = int(linedict['pad_w'])

		if layerdict['type'] == "InnerProduct":
			if 'num_output' in layerlines[i]:
				layerdict['num_output'] = int(linedict['num_output'])

	return layerdict

def count_layer_info(layerstrs, args):
	
	output_h = args.height
	output_w = args.width
	output_c = 3
	params_total = 0
	for i in range(1, len(layerstrs)):
		layerdict = parse_layer(layerstrs[i])
		if len(layerdict) == 1:
			print layerdict['name'] 
			print 'INPUT:{}x{}x3'.format(args.width, args.height)
			continue

		if 'Data' in layerdict['type']:
			continue

		if layerdict['type'] == 'Convolution':
			input_h = output_h
			input_w = output_w
			input_c = output_c
			kernel_ext_w = 2*np.floor(layerdict['kernel_w']/2)
			kernel_ext_h = 2*np.floor(layerdict['kernel_h']/2)
			output_w = np.floor((input_w + 2*layerdict['pad_w'] - kernel_ext_w)/layerdict['stride_w'])+1
			output_h = np.floor((input_h + 2*layerdict['pad_h'] - kernel_ext_h)/layerdict['stride_h'])+1
			output_c = layerdict['num_output']
			params_conv = layerdict['kernel_w']*layerdict['kernel_h']*input_c*output_c
			params_total = params_total + params_conv
			print colored('CONV({}): [{}x{}x{}]'.format(layerdict['name'], int(output_w), int(output_h), output_c), 'red')
			print colored('params: ({}x{}x{})x{} = {}'.format(layerdict['kernel_w'], layerdict['kernel_h'], input_c, output_c, \
				params_conv), 'blue')


		if layerdict['type'] == 'Pooling':
			input_h = output_h
			input_w = output_w
			input_c = output_c
			kernel_ext_w = 2*np.floor(layerdict['kernel_w']/2)
			kernel_ext_h = 2*np.floor(layerdict['kernel_h']/2)
			output_w = np.floor((input_w + 2*layerdict['pad_w'] - kernel_ext_w)/layerdict['stride_w'])+1
			output_h = np.floor((input_h + 2*layerdict['pad_h'] - kernel_ext_h)/layerdict['stride_h'])+1
			# output_c keep unchanged
			print colored('POOL({}): [{}x{}x{}]'.format(layerdict['name'], int(output_w), int(output_h), output_c), 'red')
			print colored('params: 0', 'blue')

		if layerdict['type'] == 'InnerProduct':
			input_h = output_h
			input_w = output_w
			input_c = output_c
			output_w = 1
			output_h = 1
			output_c = layerdict['num_output']
			print colored('FC({}): [{}x{}x{}]'.format(layerdict['name'], output_w, output_h, output_c), 'red')
			input_feat_dim = int(input_h*input_w*input_c)
			output_feat_dim = int(output_h*output_w*output_c)
			params_fc = input_feat_dim*output_feat_dim
			params_total = params_total + params_fc
			print colored('params: {}x{} = {}'.format(input_feat_dim, output_feat_dim, params_fc), 'blue')

	print colored('TOTAL params: %.1f MB parameters'% (params_total*4.0/1024/1024), 'blue')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--prototxt', help='caffe prototxt of model definition', required=True)
	parser.add_argument('-w', '--width', help='input image width', type=int, required=True)
	parser.add_argument('-g', '--height', help='input image height', type=int, required=True)
	args = parser.parse_args()

	f = open(args.prototxt, 'r')
	prototxt = f.read()
	f.close()

	if "layers" not in prototxt:
		layerstrs = prototxt.split('layer {')
		for i in range(1, len(layerstrs)):
			layerstrs[i] = 'layer {'+layerstrs[i]
	else:
		layerstrs = prototxt.split('layers {')
		for i in range(1, len(layerstrs)):
			layerstrs[i] = 'layers {'+layerstrs[i]

	count_layer_info(layerstrs, args)

if __name__ == "__main__":
	main()
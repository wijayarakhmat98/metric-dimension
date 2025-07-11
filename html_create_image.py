#!/usr/bin/env python3.13

from functools import partial
import multiprocessing
import os
import sys
import utils

def graph6_draw_forward(s, root_output):
	hash = utils.hash(s)
	filename = '{}/{}.svg'.format(root_output, hash)
	utils.graph6_draw(s, filename)

def main(args):
	if len(args) != 2:
		print('\t<graph6 filename> <output root>')
		return

	filename_graph6 = args[0]
	root_output = args[1]

	if not os.path.exists(root_output):
		os.makedirs(root_output)

	list_graph6 = utils.file_to_list(filename_graph6)
	with multiprocessing.Pool() as pool:
		bound_graph6_draw = partial(graph6_draw_forward, root_output=root_output)
		list(pool.imap_unordered(bound_graph6_draw, list_graph6))

if __name__ == '__main__':
	main(sys.argv[1:])

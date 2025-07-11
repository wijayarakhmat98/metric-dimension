#!/usr/bin/env python3.13

import ast
from functools import partial
import multiprocessing
import os
import sys
import utils

def info_create(r, root_output, template_html, filename_css):
	r = ast.literal_eval(r)
	hash = utils.hash(r['graph'])
	filename = '{}/{}.html'.format(root_output, hash)
	with open(filename, 'w') as file:
		file.write(template_html.format(
			filename_css,
			r['graph'],
			'{}.svg'.format(hash),
			r['graph'],
			utils.info_stringify(r)
		))

def main(args):
	if len(args) != 2:
		print('\t<result filename> <output root>')
		return

	filename_result = args[0]
	root_output = args[1]
	filename_css = 'info.css'

	if not os.path.exists(root_output):
		os.makedirs(root_output)

	with open('{}/{}'.format(root_output, filename_css), 'w') as file:
		file.write(template_css)

	list_result = utils.file_to_list(filename_result)
	with multiprocessing.Pool() as pool:
		bound_info_create = partial(info_create, root_output=root_output, template_html=template_html, filename_css=filename_css)
		list(pool.imap_unordered(bound_info_create, list_result))

template_html = '''
<!DOCTYPE html>
<html>
	<head>
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<link rel="stylesheet" href="{}">
		<title>{}</title>
	</head>
	<body>
		<img src="{}" alt="{}">
		<pre>{}</pre>
	</body>
</html>
'''

template_css = '''
body {
	display: flex;
	flex-wrap: wrap;
	justify-content: center;
	align-items: start;
	gap: 1em;
	margin: 1em;
}

body * {
	border: 1px solid black;
	margin: 0;
	padding: 1em;
}

img {
	max-width: 15em;
}
'''

if __name__ == '__main__':
	main(sys.argv[1:])

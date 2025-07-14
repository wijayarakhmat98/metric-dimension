#!/usr/bin/env python3.12

import ast
from functools import partial
import multiprocessing
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils

def figure_create(r, template_html_figure):
	r = ast.literal_eval(r)
	hash = utils.hash(r['graph'])
	return template_html_figure.format(
		'{}.svg'.format(hash),
		'{}.html'.format(hash),
		r['graph'],
		r['vertex']['dimension'],
		r['edge']['dimension']
	)

def main(args):
	if len(args) != 2:
		print('\t<result filename> <output root>')
		return

	filename_result = args[0]
	root_output = args[1]
	filename_html = 'index.html'
	filename_css = 'index.css'
	name = re.sub(r'[^/]*/', '', filename_result)
	name = re.sub(r'\.[^.]*$', '', name)

	if not os.path.exists(root_output):
		os.makedirs(root_output)

	with open('{}/{}'.format(root_output, filename_css), 'w') as file:
		file.write(template_css)

	list_result = utils.file_to_list(filename_result)
	with open('{}/{}'.format(root_output, filename_html), 'w') as file:
		file.write(template_html_open.format(filename_css, name))
		with multiprocessing.Pool() as pool:
			bound_figure_create = partial(figure_create, template_html_figure=template_html_figure)
			for figure in pool.imap(bound_figure_create, list_result):
				file.write(figure)
		file.write(template_html_close)

template_html_open = '''
<!DOCTYPE html>
<html>
	<head>
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<link rel="stylesheet" href="{}">
		<title>{}</title>
	</head>
	<body>
'''

template_html_close = '''
	</body>
</html>
'''

template_html_figure = '<div><img src={} loading=lazy><p><a href={}>{}</a><br>{}m {}e</p></div>'

template_css = '''
body {
	display: flex;
	flex-wrap: wrap;
	justify-content: center;
	align-items: start;
	gap: 1em;
	margin: 1em;
}

body > * {
	border: 1px solid black;
	margin: 0;
	padding: 1em;
}

body > div {
	width: 15em;
	height: 18em;
	display: flex;
	flex-direction: column;
	align-items: center;
	gap: 1em;
}

body > div > img {
	flex: 1 1 0;
	min-height: 0;
	max-width: 100%;
}

body > div > p {
	flex: 0 0 auto;
	text-align: center;
	margin: 0;
}

body > div > p > a {
	font-weight: bold;
	text-decoration: none;
}
'''

if __name__ == '__main__':
	main(sys.argv[1:])

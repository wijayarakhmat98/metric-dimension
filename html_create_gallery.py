#!/usr/bin/env python3.13

import ast
import json
import multiprocessing
import os
import re
import sys
import utils

def ast_read(r):
	r = ast.literal_eval(r)
	return [
		r['graph'],
		r['vertex']['dimension'],
		r['edge']['dimension']
	]

def main(args):
	if len(args) != 2:
		print('\t<result filename> <output root>')
		return

	filename_result = args[0]
	root_output = args[1]
	filename_json = 'result.js'
	filename_html = 'index.html'
	filename_css = 'index.css'
	filename_js = 'index.js'
	name = re.sub(r'[^/]*/', '', filename_result)
	name = re.sub(r'\.[^.]*$', '', name)

	if not os.path.exists(root_output):
		os.makedirs(root_output)

	with open('{}/{}'.format(root_output, filename_css), 'w') as file:
		file.write(template_css)

	with open('{}/{}'.format(root_output, filename_js), 'w') as file:
		file.write(template_js)

	with open('{}/{}'.format(root_output, filename_html), 'w') as file:
		file.write(template_html.format(
			filename_css,
			filename_json,
			filename_js,
			filename_json,
			filename_js,
			name
		))

	list_result = utils.file_to_list(filename_result)
	with multiprocessing.Pool() as pool:
		list_result = list(pool.imap(ast_read, list_result))

	with open('{}/{}'.format(root_output, filename_json), 'w') as file:
		file.write('const result = ')
		file.write(json.dumps(list_result))
		file.write('\nexport default result')

template_html = '''
<!DOCTYPE html>
<html>
	<head>
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<link rel="stylesheet" href="{}">
		<link rel="modulepreloaded" href="{}">
		<link rel="modulepreloaded" href="{}">
		<script type="importmap">
			{{
				"imports": {{
					"result": "./{}",
					"main": "./{}"
				}}
			}}
		</script>
		<script type="module">
			import result from 'result'
			import main from 'main'
			main(result)
		</script>
		<title>{}</title>
	</head>
	<body>
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

body > * {
	border: 1px solid black;
	margin: 0;
	padding: 1em;
}

figure {
	width: 15em;
	height: 18em;
	display: flex;
	flex-direction: column;
	align-items: center;
	gap: 1em;
}

figure img {
	flex: 1 1 0;
	min-height: 0;
	max-width: 100%;
}

figcaption {
	flex: 0 0 auto;
	text-align: center;
}

figcaption a {
	font-weight: bold;
	text-decoration: none;
}
'''

template_js = '''
const template_figure = `
	<img src="{src}" alt="{alt}" loading="lazy">
	<figcaption>
		<a href="{info}">{graph}</a><br>
		Metric dimension: {d_vertex}<br>
		Edge dimension: {d_edge}
	</figcaption>
`

async function sha256(str) {
	const encoder = new TextEncoder();
	const data = encoder.encode(str);
	const hashBuffer = await crypto.subtle.digest('SHA-256', data);
	const hashArray = Array.from(new Uint8Array(hashBuffer));
	return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

async function main(result) {
	for (const [graph, d_vertex, d_edge] of result) {
		const hash = await sha256(graph)
		const f = document.createElement('figure')
		f.innerHTML = template_figure
			.replace('{src}', `${hash}.svg`)
			.replace('{alt}', graph)
			.replace('{info}', `${hash}.html`)
			.replace('{graph}', graph)
			.replace('{d_vertex}', d_vertex)
			.replace('{d_edge}', d_edge)
		document.body.appendChild(f)
	}
}

export default main
'''

if __name__ == '__main__':
	main(sys.argv[1:])

#!/usr/bin/env python3.13

import ast
import multiprocessing
import os
import re
import shutil
import sys
import utils

html_template_index = '''
<!DOCTYPE html>
<html>
	<head>
		<title>Dimension Metric</title>
	</head>
	<body>
		<h1>Dimension Metric</h1>
		<ul>
			{}
		</ul>
	</body>
</html>
'''

css_template_figure = '''
.figure {
	display: flex;
	flex-wrap: wrap;
	justify-content: center;
}

.figure figure {
	width: 15em;
	height: 18em;
	display: flex;
	flex-direction: column;
	align-items: center;
	border: 1px solid black;
	margin: 1em;
}

.figure figure div {
	flex: 1 1 auto;
	display: flex;
	align-items: center;
	justify-content: center;
	width: 100%;
	overflow: hidden;
	margin: 0.5em;
}

.figure figure div img {
	max-width: 100%;
	max-height: 100%;
	object-fit: contain;
}

.figure figure figcaption {
	text-align: center;
	flex: 0 0 auto;
	margin: 0.5em;
}

.figure figure figcaption a {
	text-decoration: none;
}
'''

html_template_gallery = '''
<!DOCTYPE html>
<html>
	<head>
		<title>{}</title>
		<link rel="stylesheet" href="{}">
	</head>
	<body>
		<div class="figure">
			{}
		</div>
	</body>
</html>
'''

html_template_gallery_figure = '''
<figure>
	<div>
		<img src="{}">
	</div>
	<figcaption>
		<a href="{}"><strong>{}</strong><br></a>
		Metric dimension: {}<br>
		Edge dimension: {}
	</figcaption>
</figure>
'''

html_template_info = '''
<!DOCTYPE html>
<html>
	<head>
		<title>{}</title>
		<link rel="stylesheet" href="{}">
		<style>
			.column {{
				display: flex;
				justify-content: center;
				gap: 1em;
			}}

			.column div {{
				display: inline-block;
			}}
		</style>
	</head>
	<body>
		<div class="column">
			<div class="figure">
				{}
			</div>
			<div>
				<pre>{}</pre>
			</div>
		</div>
	</body>
</html>
'''

html_template_info_figure = '''
<figure>
	<div>
		<img src="{}">
	</div>
	<figcaption>
		<strong>{}</strong><br>
		Metric dimension: {}<br>
		Edge dimension: {}
	</figcaption>
</figure>
'''

def graph6_draw_forward(args):
	utils.graph6_draw(*args)

def main(args):
	root_html = 'docs'
	if os.path.exists(root_html):
		shutil.rmtree(root_html)
	os.makedirs(root_html)

	filename_index = 'index.html'
	html_body_index = ''

	filename_style_figure = 'figure.css'

	for filename_data in args:
		name = re.sub(r'^.*/(.*)\.[^.]*$', r'\1', filename_data)

		root_gallery = name
		os.makedirs('{}/{}'.format(root_html, root_gallery))

		root_img = 'img'
		os.makedirs('{}/{}/{}'.format(root_html, root_gallery, root_img))

		root_info = 'info'
		os.makedirs('{}/{}/{}'.format(root_html, root_gallery, root_info))

		data = [ast.literal_eval(s) for s in utils.file_to_list(filename_data)]
		for d in data:
			d['hash'] = utils.hash(d['graph'])
			d['filename_img'] = '{}/{}.svg'.format(root_img, d['hash'])
			d['filename_info'] = '{}/{}.html'.format(root_info, d['hash'])

		with multiprocessing.Pool() as pool:
			list(pool.imap_unordered(graph6_draw_forward, [(d['graph'], '{}/{}/{}'.format(root_html, root_gallery, d['filename_img'])) for d in data]))

		filename_gallery = '{}/index.html'.format(root_gallery)
		html_body_gallery = ''

		html_body_index += '<li><a href="{}">{}</a></li>'.format(filename_gallery, name)

		for d in data:
			html_body_gallery += html_template_gallery_figure.format(
				d['filename_img'],
				d['filename_info'],
				d['graph'],
				d['vertex']['dimension'],
				d['edge']['dimension']
			)

			with open('{}/{}/{}'.format(root_html, root_gallery, d['filename_info']), 'w') as file_info:
				dd = d.copy()
				del dd['hash']
				del dd['filename_img']
				del dd['filename_info']
				file_info.write(html_template_info.format(
					d['graph'],
					'../../{}'.format(filename_style_figure),
					html_template_info_figure.format(
						'../{}'.format(d['filename_img']),
						d['graph'],
						d['vertex']['dimension'],
						d['edge']['dimension']
					),
					utils.info_stringify(dd)
				))

		with open('{}/{}'.format(root_html, filename_gallery), 'w') as file_gallery:
			file_gallery.write(html_template_gallery.format(name, '../{}'.format(filename_style_figure), html_body_gallery))

	with open('{}/{}'.format(root_html, filename_index), 'w') as file_index:
		file_index.write(html_template_index.format(html_body_index))

	with open('{}/{}'.format(root_html, filename_style_figure), 'w') as file_style_figure:
		file_style_figure.write(css_template_figure)

if __name__ == '__main__':
	main(sys.argv[1:])

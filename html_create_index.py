#!/usr/bin/env python3.13

import os
import sys

def main(args):
	if len(args) != 1:
		print('\t<output root>')
		return

	root_output = args[0]
	filename_index = 'index.html'

	if not os.path.exists(root_output):
		os.makedirs(root_output)

	list_page = []
	for entry in os.listdir(root_output):
		path = os.path.join(root_output, entry)
		if os.path.isdir(path):
			list_page.append(entry)
	list_page.sort()

	with open('{}/{}'.format(root_output, filename_index), 'w') as file:
		file.write(template_html.format('\n'.join([
			'<li><a href="{}">{}</a></li>'.format(entry, entry)
			for entry in list_page
		])))

template_html = '''
<!DOCTYPE html>
<html>
	<head>
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<title>Metric Dimension</title>
	</head>
	<body>
		<h1>Metric Dimension</h1>
		<ul>
			{}
		</ul>
	</body>
</html>
'''

if __name__ == '__main__':
	main(sys.argv[1:])

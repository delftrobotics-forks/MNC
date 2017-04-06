import argparse
import os
import sys
import yaml

from jinja2 import Environment, FileSystemLoader

def renderTemplate(env, template_file, context):
	return env.get_template(template_file).render(context)

def createPrototxtFile(env, template_file, context, prototxt_file):
	with open(prototxt_file, "w") as f:
		f.write(renderTemplate(env, template_file, context))

def readYamlFile(yaml_file):
	with open(yaml_file, "r") as f:
		return yaml.load(f)

def run(args):
	env = Environment(
		autoescape  = False,
		loader      = FileSystemLoader("templates"),
		trim_blocks = False
	)

	context = {}
	if args.yaml_files:
		for y in args.yaml_files:
			try:
				context.update(readYamlFile(y))
			except IOError:
				print("Unable to open parameters file '%s'." % y)
				sys.exit(-1)

	if args.parameters:
		for p in args.parameters:
			name, sep, value = p.partition("=")
			if sep == "=": context[name] = value

	createPrototxtFile(env, args.template_file, context, args.prototxt_file)
	print("Generated prototxt file: '%s'" % args.prototxt_file)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Generate train/test prototxt files for the MNC network.")
	parser.add_argument("template_file",      help = "path to the template file which stores the network architecture")
	parser.add_argument("prototxt_file",      help = "path to the output prototxt file")
	parser.add_argument("-y", "--yaml_files", help = "path to the YAML files which stores the parameters",       action = "append")
	parser.add_argument("-p", "--parameters", help = "overrides certain parameters specified in the YAML files", action = "append")
	args = parser.parse_args()

	if not os.path.exists(os.path.join(os.getcwd(), "templates")):
		print("Please run this script from the MNC working directory.")
		sys.exit(-1)

	run(args)


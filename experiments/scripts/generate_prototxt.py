import os
import sys
import yaml

from jinja2 import Environment, FileSystemLoader

env = Environment(
  autoescape  = False,
  loader      = FileSystemLoader(os.path.join("templates")),
  trim_blocks = False
)

def renderTemplate(template_file, context):
  return env.get_template(template_file).render(context)

def createPrototxtFile(template_file, context, prototxt_file):
  with open(prototxt_file, "w") as f:
    f.write(renderTemplate(template_file, context))

def readYamlFile(yaml_file):
  with open(yaml_file, "r") as f:
    return yaml.load(f)

def run(yaml_file, template_file, prototxt_file):
  context = readYamlFile(yaml_file)
  createPrototxtFile(template_file, context, prototxt_file)
  print("Saved to: '%s'" % prototxt_file)

if __name__ == "__main__":
  if len(sys.argv) < 4:
    print("Usage:", sys.argv[0], "[yaml_file] [template_file] [prototxt_file]")
    sys.exit(-1)

  run(sys.argv[1], sys.argv[2], sys.argv[3])


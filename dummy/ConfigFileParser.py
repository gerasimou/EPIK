from configparser import ConfigParser

config = ConfigParser()
config.read_file(open(r'configFIle.txt'))
path1 = config.get('My Section', 'path1')
path2 = config.get('My Section', 'path2')
path3 = config.get('My Section', 'path3')


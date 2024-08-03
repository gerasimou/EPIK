from configparser import ConfigParser, NoOptionError


properties_file = 'configFile.txt'
config = ConfigParser()
config.read_file(open(properties_file))

try:
    #Independent Runs
    RUNS = config.getint('EPIK_Properties', 'RUNS')

    #Problem name
    PROBLEM_NAME = config.get('EPIK_Properties', 'PROBLEM_NAME')

    #Number of Samples
    n = config.getint('EPIK_Properties', 'n')

    #Number of variables to learn (2 per distribution)
    NVARS = config.getint('EPIK_Properties', 'NVARS')

    #GA Population
    POPULATION = config.getint('EPIK_Properties', 'POPULATION')

    #Number of generations
    GENERATIONS = config.getint('EPIK_Properties', 'GENERATIONS')

    #Folder to store the results
    DATA_DIR = config.get('EPIK_Properties', 'DATA_DIR')

    # Conformance level of prior knowledge between R1 and R2
    # 0->full conformance, 1->light conformance, 2-> light conflict, 3-> heavy conflict
    CONFORMANCE_LEVEL = config.getint('EPIK_Properties', 'CONFORMANCE_LEVEL')


    #Model type
    MODEL_TYPE = config.get('EPIK_Properties', 'MODEL_TYPE')
except NoOptionError as e:
    print("ERROR:", e.message)


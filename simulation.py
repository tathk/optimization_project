import time
import gc
import os
import sys

from collections import defaultdict
import cplex


def get_dataset_name(filename):
    filename = '/'.join(filename.split('/')[-2:])
    return filename.split('.')[0]  

# return dictionary of all details
def parse_solomon_to_distance_matrix(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if line.strip() and not line.startswith(';')]
        vehicle_number, vehicle_capacity = map(int, lines[3].split())
        # print(f"Vehicle number: {vehicle_number}, Vehicle capacity: {vehicle_capacity}")
        customers_list = lines[6::]

        customer_dict = {}

        for customer_line in customers_list:
            customer_id, x, y, demand, ready_time, due_date, service = map(int, customer_line.split())
            customer_dict[customer_id] = {
                'x': x,
                'y': y,
                'demand': demand,
                'ready_time': ready_time,
                'due_date': due_date,
                'service': service
            }          

        depot = customer_dict[0]
        depot_end = depot.copy()
        customer_dict[len(customer_dict)] = depot_end

    return vehicle_number, vehicle_capacity, customer_dict  
def calculate_distance_matrix(customers_dict):
    customer_ids = list(customers_dict.keys())
    n = len(customer_ids)
    distance_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = customers_dict[customer_ids[i]]['x'], customers_dict[customer_ids[i]]['y']
                x2, y2 = customers_dict[customer_ids[j]]['x'], customers_dict[customer_ids[j]]['y']
                distance_matrix[i][j] = (((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

    return distance_matrix



from docplex.mp.model import Model

def create_crvptmodel(file_name, optimal_vehicle_number = None):
    vehicle_number, vehicle_capacity, node_dict = parse_solomon_to_distance_matrix(file_name)

    distance_matrix = calculate_distance_matrix(node_dict)
    # set of customers including depot
    V = list(node_dict.keys()) 
    # print("V:", V)
    # set of customers excluding depot
    N = V[1:-1]


    
    # set of vehicles
    # if optimal vehcile is provide use it, otherwise go with max
    if optimal_vehicle_number is not None:
        vehicle_number = optimal_vehicle_number
    K = range(vehicle_number)  # vehicle_number


    start_depot = 0
    end_depot = max(V)  # ID of the duplicated depot
    N = [i for i in V if i not in [start_depot, end_depot]]  # Only real customers

    

    # distance matrix
    c = distance_matrix
    m = Model(name='Solomon CRVPT')
    x = m.binary_var_dict(((i, j, k) for i in V for j in V for k in K if i != j), name="x")
    T = m.continuous_var_dict(((i, k) for i in V for k in K), name="T")

    # objective function
    m.minimize(
        m.sum(c[i][j] * x[i, j, k] for i in V for j in V if i != j for k in K)
    )

    # Constraints: Each customer is visited exactly once by one vehicle
    for i in N:
        m.add_constraint(
            m.sum(x[i, j, k] for k in K for j in V if j != i) == 1
        )

    # Constraints: Each vehicle leaves the depot once
    # if optimal vehicle is provided, then == 1 to force every vehicle to leave the depot, otherwise find them agressively
    if optimal_vehicle_number is not None:
        for k in K:
            m.add_constraint(
                m.sum(x[start_depot, j, k] for j in N) == 1
            )
    else:
        for k in K:
            m.add_constraint(
                m.sum(x[start_depot, j, k] for j in N) <= 1
            )

    # flow conservation constraints 5.4)
    for k in K:
        for j in N:
            m.add_constraint(
                m.sum(x[i, j, k] for i in V if i != j) ==
                m.sum(x[j, i, k] for i in V if i != j)
            )

    # Constraints: Each vehicle returns to the depot 5.5)
    if optimal_vehicle_number is not None:
        for k in K:
            m.add_constraint(
                m.sum(x[i, end_depot, k] for i in N) == 1
            )
    else:
        for k in K:
            m.add_constraint(
                m.sum(x[i, end_depot, k] for i in N) <= 1
            )

    # time consistency constraints
    for k in K:
        for i in V:
            for j in V:
                if i != j:
                    m.add_indicator(
                        x[i, j, k],
                        T[i, k] + node_dict[i]['service'] + c[i][j] <= T[j, k]
                    )

    # time window constraints
    for k in K:
        for i in V:
            m.add_constraint(T[i, k] >= node_dict[i]['ready_time'])
            m.add_constraint(T[i, k] <= node_dict[i]['due_date'])

    # vehicle capacity constraints
    for k in K:
        m.add_constraint(
            m.sum(node_dict[i]['demand'] * x[i, j, k] for i in N for j in V if i != j) <= vehicle_capacity
        )

    # solution = m.solve(log_output=True)
    m.context.cplex_parameters.timelimit = 120  # seconds
    m.context.cplex_parameters.threads = 1  
    start_time = time.time()
    solution = m.solve(log_output=True)
    end_time = time.time()
    

    if not solution:
        # print("No solution found.")
        print("Solve status:", m.solve_status)
        gc.collect()
        return
    else:
        print("Objective value:", m.objective_value, flush=True)
        objective_value = m.objective_value
        m.end()
        gc.collect()
        return m, x, T, (end_time - start_time) , objective_value


def get_routes_from_solution(x):
    def reconstruct_route(routes, vehicle_id):
        route = []
        current_node = 0  
        while True:
            next_node = None
            for (i, j) in routes[vehicle_id]:
                if i == current_node:
                    next_node = j
                    break
            if next_node is None or next_node == 0:  
                break
            route.append((current_node, next_node))
            current_node = next_node
        return route

    routes = defaultdict(list)
    ret_routes = defaultdict(list)
    for (i, j, k), var in x.items():
        if var.solution_value == 1:
            routes[k].append((i, j))

    for vehicle_id in routes:
        ret_routes[vehicle_id] = reconstruct_route(routes, vehicle_id)
    return list(ret_routes.values())


def simulation(filename, optimal_given=None):
    try:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        
        if optimal_given is None:
            outputfilename = "output/" + "optimal_not_given/" + get_dataset_name(filename) 
            os.makedirs(os.path.dirname(outputfilename), exist_ok=True)

            m, x, T, time, obj_value = create_crvptmodel(filename)
            with open(outputfilename, 'a') as f:
                f.write(f"{len(x)} {len(T)} {len(x)+len(T)} {time} {obj_value}\n")
            
        else:
            outputfilename = "output/" + "optimal_given/" + get_dataset_name(filename) 
            os.makedirs(os.path.dirname(outputfilename), exist_ok=True)
            m, x, T, time, obj_value = create_crvptmodel(filename, optimal_vehicle_number=optimal_given)
            with open(outputfilename, 'a') as f:
                f.write(f"{len(x)} {len(T)} {len(x)+len(T)} {time} {obj_value}\n")
    except Exception as e:
        print(f"Error while processing {filename}: {e}")


if __name__ == "__main__":
    filename = sys.argv[1]
    simulation(filename)

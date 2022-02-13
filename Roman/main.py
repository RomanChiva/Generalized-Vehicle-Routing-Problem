from prob_form import problem_formulation as problem_formulation

problem = problem_formulation('dataproblem1.xlsx',1,400,1)
problem.cluster(3)
problem.demand(random=True, u = 5, l=1)
problem.add_origin_depot(50,50)
problem.decision_variables()
problem.degree_constraints()
problem.flow_constraints()
problem.side_and_subtour_elimination()
problem.add_obj()
problem.optimize_model()
#problem.display_results('dghkdvs')
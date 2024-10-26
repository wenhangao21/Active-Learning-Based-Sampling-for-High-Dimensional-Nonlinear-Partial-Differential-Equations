import numpy
import matplotlib

# generate uniform distributed points in a domain [a1,b1]X[a2,b2]X...X[ad,bd]
# domain_intervals = [[a1,b1],[a2,b2],...,[ad,bd]]
def generate_uniform_points_in_cube(domain_intervals,N,if_seed=False):
    if if_seed == True:
        numpy.random.seed(1)
    d = domain_intervals.shape[0]
    points = numpy.zeros((N,d))
    for i in range(d):
        points[:,i] = numpy.random.uniform(domain_intervals[i,0],domain_intervals[i,1],(N,))
    return points

# generate uniform distributed points in a domain [T0,T1]X[a1,b1]X[a2,b2]X...X[ad,bd]
# domain_intervals = [[a1,b1],[a2,b2],...,[ad,bd]]
def generate_uniform_points_in_cube_time_dependent(domain_intervals,time_interval,N,if_seed=False):
    if if_seed == True:
        numpy.random.seed(1)
    d = domain_intervals.shape[0]
    points = numpy.zeros((N,d+1))
    points[:,0] = numpy.random.uniform(time_interval[0],time_interval[1],(N,))
    for i in range(0,d):
        points[:,i+1] = numpy.random.uniform(domain_intervals[i,0],domain_intervals[i,1],(N,))
    return points

# generate uniform distributed points in the sphere {x:|x|<R}
# input d is the dimension
def generate_uniform_points_in_sphere(d,R,N,if_seed=False):
    if if_seed == True:
        numpy.random.seed(1)
    points = numpy.random.normal(size=(N,d))
    scales = (numpy.random.uniform(0,1,(N,)))**(1/d)
    for i in range(N):
        points[i,:] = points[i,:]/numpy.sqrt(numpy.sum(points[i,:]**2))*scales[i]*R
    return points

# generate uniform distributed points in the sphere [T0,T1]X{x:|x|<R}
# input d is the dimension
def generate_uniform_points_in_sphere_time_dependent(d,R,time_interval,N,if_seed=False):
    if if_seed == True:
        numpy.random.seed(1)
    points = numpy.zeros((N,d+1))
    points[:,0] = numpy.random.uniform(time_interval[0],time_interval[1],(N,))
    points[:,1:] = numpy.random.normal(size=(N,d))
    scales = (numpy.random.uniform(0,1,(N,)))**(1/d)
    for i in range(N):
        points[i,1:] = points[i,1:]/numpy.sqrt(numpy.sum(points[i,1:]**2))*scales[i]*R
    return points

# generate uniform distributed points in the annulus {x:r<|x|<R}
# input d is the dimension
def generate_uniform_points_in_annulus(d,r,R,N,if_seed=False):
    if if_seed == True:
        numpy.random.seed(1)
    points = numpy.random.normal(size=(N,d))
    scales = (numpy.random.uniform(r**d,R**d,(N,)))**(1/d)
    for i in range(N):
        points[i,:] = points[i,:]/numpy.sqrt(numpy.sum(points[i,:]**2))*scales[i]
    return points

# generate points in the sphere {x:|x|<R}
# these points are uniformly distributed in each annulus {x:kR/N1<x<(k+1)R/N1}, k=0,1,...,N1-1
# Every annulus has N2 points
# input d is the dimension
def generate_uniform_annular_points_in_sphere(d,R,N1,N2,if_seed=False):
    if if_seed == True:
        numpy.random.seed(1)
    points = numpy.zeros((N1*N2,d))
    for k in range(N1):
        R_outer = (k+1)*R/N1
        R_inner = k*R/N1
        temp_pts = numpy.random.normal(size=(N2,d))
        scales = (numpy.random.uniform(R_inner**d,R_outer**d,(N2,)))**(1/d)
        for i in range(N2):
            temp_pts[i,:] = temp_pts[i,:]/numpy.sqrt(numpy.sum(temp_pts[i,:]**2))*scales[i]
        points[k*N2:(k+1)*N2,:] = temp_pts
    return points

# generate points in the sphere [T0,T1]X{x:|x|<R}
# these points are uniformly distributed in each annulus [T0,T1]X{x:kR/N1<x<(k+1)R/N1}, k=0,1,...,N1-1
# Every annulus has N2 points
# input d is the dimension
def generate_uniform_annular_points_in_sphere_time_dependent(d,R,time_interval,N1,N2,if_seed=False):
    if if_seed == True:
        numpy.random.seed(1)
    points = numpy.zeros((N1*N2,d+1))
    points[:,0] = numpy.random.uniform(time_interval[0],time_interval[1],(N1*N2,))
    for k in range(N1):
        R_outer = (k+1)*R/N1
        R_inner = k*R/N1
        temp_pts = numpy.random.normal(size=(N2,d))
        scales = (numpy.random.uniform(R_inner**d,R_outer**d,(N2,)))**(1/d)
        for i in range(N2):
            temp_pts[i,:] = temp_pts[i,:]/numpy.sqrt(numpy.sum(temp_pts[i,:]**2))*scales[i]
        points[k*N2:(k+1)*N2,1:] = temp_pts
    return points

# generate uniform distributed points on the boundary of domain [a1,b1]X[a2,b2]X...X[ad,bd]
# domain_intervals = [[a1,b1],[a2,b2],...,[ad,bd]]
def generate_uniform_points_on_cube(domain_intervals,N_each_face,if_seed=False):
    if if_seed == True:
        numpy.random.seed(1)
    d = domain_intervals.shape[0]
    if d == 1:
        return numpy.array([[domain_intervals[0,0]],[domain_intervals[0,1]]])
    else:
        points = numpy.zeros((2*d*N_each_face,d))
        for i in range(d):
            points[2*i*N_each_face:(2*i+1)*N_each_face,:] = numpy.insert(generate_uniform_points_in_cube(numpy.delete(domain_intervals,i,axis=0),N_each_face), i, values=domain_intervals[i,0]*numpy.ones((1,N_each_face)), axis = 1)
            points[(2*i+1)*N_each_face:(2*i+2)*N_each_face,:] = numpy.insert(generate_uniform_points_in_cube(numpy.delete(domain_intervals,i,axis=0),N_each_face), i, values=domain_intervals[i,1]*numpy.ones((1,N_each_face)), axis = 1)
        return points

# generate uniform distributed points on the boundary of time-dependent domain [T0,T1]X[a1,b1]X[a2,b2]X...X[ad,bd]
# and at the initial slice slice {t=T0}X[a1,b1]X[a2,b2]X...X[ad,bd]
# whole_intervals = [[T0,T1]X[a1,b1],[a2,b2],...,[ad,bd]]
def generate_uniform_points_on_cube_time_dependent(domain_intervals,time_interval,N_each_face,N_initial_time_slice, time_condition_type = 'initial',if_seed=False):
    if if_seed == True:
        numpy.random.seed(1)
    d = domain_intervals.shape[0]
    whole_intervals = numpy.zeros((d+1,2))
    whole_intervals[0,:] = time_interval
    whole_intervals[1:,:] = domain_intervals
    points_bd = numpy.zeros((2*d*N_each_face,1+d))
    points_int = numpy.zeros((N_initial_time_slice,1+d))
    for i in range(1,1+d):
        points_bd[2*(i-1)*N_each_face:(2*i-1)*N_each_face,:] = numpy.insert(generate_uniform_points_in_cube(numpy.delete(whole_intervals,i,axis=0),N_each_face), i, values=whole_intervals[i,0]*numpy.ones((1,N_each_face)), axis = 1)
        points_bd[(2*i-1)*N_each_face:2*i*N_each_face,:] = numpy.insert(generate_uniform_points_in_cube(numpy.delete(whole_intervals,i,axis=0),N_each_face), i, values=whole_intervals[i,1]*numpy.ones((1,N_each_face)), axis = 1)
    if time_condition_type == 'initial':
        points_int = numpy.insert(generate_uniform_points_in_cube(numpy.delete(whole_intervals,0,axis=0),N_initial_time_slice), 0, values=whole_intervals[0,0]*numpy.ones((1,N_initial_time_slice)), axis = 1)
    elif time_condition_type == 'terminal':
        points_int = numpy.insert(generate_uniform_points_in_cube(numpy.delete(whole_intervals,0,axis=0),N_initial_time_slice), 0, values=whole_intervals[0,1]*numpy.ones((1,N_initial_time_slice)), axis = 1)
    return points_bd, points_int

# generate uniform distributed points on the boundary of domain {|x|<R}
def generate_uniform_points_on_sphere(d,R,N_boundary,if_seed=False):
    if if_seed == True:
        numpy.random.seed(1)
    if d == 1:
        return numpy.array([[-R],[R]])
    else:
        points = numpy.zeros((N_boundary,d))
        for i in range(N_boundary):
            points[i,:] = numpy.random.normal(size=(1,d))
            points[i,:] = points[i,:]/numpy.sqrt(numpy.sum(points[i,:]**2))*R
        return points
    
# generate uniform distributed points on the boundary of domain {|x|<R}
def generate_uniform_points_on_annulus(d,r,R,N_boundary,if_seed=False):
    if if_seed == True:
        numpy.random.seed(1)
    if d == 1:
        return numpy.array([[-R],[R],[-r],[r]])
    else:
        points = numpy.zeros((N_boundary,d))
        for i in range(int(numpy.floor(N_boundary/2))):
            points[i,:] = numpy.random.normal(size=(1,d))
            points[i,:] = points[i,:]/numpy.sqrt(numpy.sum(points[i,:]**2))*r
        for i in range(int(numpy.floor(N_boundary/2)),N_boundary):
            points[i,:] = numpy.random.normal(size=(1,d))
            points[i,:] = points[i,:]/numpy.sqrt(numpy.sum(points[i,:]**2))*R
        return points

# generate uniform distributed points on the boundary of time-dependent domain [T0,T1]X{|x|<R}
# except for the final slice {t=T1}X{|x|<R}
def generate_uniform_points_on_sphere_time_dependent(d,R,time_interval,N_boundary,N_initial_time_slice,if_seed=False):
    if if_seed == True:
        numpy.random.seed(1)
    if d == 1:
        return generate_uniform_points_on_cube_time_dependent(numpy.array([[-R,R]]),time_interval,int(round(N_boundary/2)), N_initial_time_slice, time_condition_type = 'initial')
    points_bd = numpy.zeros((N_boundary,d+1))
    points_int = numpy.zeros((N_initial_time_slice,d+1))
    points_int[:,0] = time_interval[0]*numpy.ones(N_initial_time_slice,)
    points_int[:,1:] = numpy.random.normal(size=(N_initial_time_slice,d))
    for i in range(N_boundary):
        points_bd[i,0] = numpy.random.uniform(time_interval[0],time_interval[1])
        points_bd[i,1:] = numpy.random.normal(size=(1,d))
        points_bd[i,1:] = points_bd[i,1:]/numpy.sqrt(numpy.sum(points_bd[i,1:]**2))*R
    points_int[:,0] = time_interval[0]*numpy.ones(N_initial_time_slice,)
    points_int[:,1:] = numpy.random.normal(size=(N_initial_time_slice,d))
    scales = (numpy.random.uniform(0,R,(N_initial_time_slice,)))**(1/d)
    for i in range(N_initial_time_slice):
        points_int[i,1:] = points_int[i,1:]/numpy.sqrt(numpy.sum(points_int[i,1:]**2))*scales[i]
    return points_bd, points_int

# generate uniform distributed points on the boundary of time-dependent domain [T0,T1]X{|x|<R}
# and uniform annular initial points (t=0) in {|x|<R}
def generate_uniform_annular_points_on_sphere_time_dependent(d,R,time_interval,N_boundary,N1,N2,if_seed=False):
    if if_seed == True:
        numpy.random.seed(1)
    if d == 1:
        return generate_uniform_points_on_cube_time_dependent(numpy.array([[-R,R]]),time_interval,int(round(N_boundary/2)), N1*N2, time_condition_type = 'initial')
    points_bd = numpy.zeros((N_boundary,d+1))
    for i in range(N_boundary):
        points_bd[i,0] = numpy.random.uniform(time_interval[0],time_interval[1])
        points_bd[i,1:] = numpy.random.normal(size=(1,d))
        points_bd[i,1:] = points_bd[i,1:]/numpy.sqrt(numpy.sum(points_bd[i,1:]**2))*R
        
    points_int = numpy.zeros((N1*N2,d+1))
    points_int[:,0] = time_interval[0]*numpy.ones(N1*N2,)
    points_int[:,1:] = numpy.zeros((N1*N2,d))
    for k in range(N1):
        R_outer = (k+1)*R/N1
        R_inner = k*R/N1
        temp_pts = numpy.random.normal(size=(N2,d))
        scales = (numpy.random.uniform(R_inner**d,R_outer**d,(N2,)))**(1/d)
        for i in range(N2):
            temp_pts[i,:] = temp_pts[i,:]/numpy.sqrt(numpy.sum(temp_pts[i,:]**2))*scales[i]
        points_int[k*N2:(k+1)*N2,1:] = temp_pts
    
    return points_bd, points_int

# generate a list of learning rates
### modified index error
def generate_learning_rates(highest_lr_pow,lowest_lr_pow,total_iterations,ratio_get_to_the_lowest,n_stage):
    lr = 10 ** lowest_lr_pow * numpy.ones((total_iterations,))
    lr_n = numpy.ceil(numpy.arange(0,n_stage+1)*(total_iterations*ratio_get_to_the_lowest/n_stage)).astype(numpy.int32)
    for i in range(n_stage):
        lr[lr_n[i]:lr_n[i+1]] = 10 ** (highest_lr_pow + (lowest_lr_pow-highest_lr_pow)/n_stage*i)
    return lr



def generate_learning_rates_high_in_the_begin(highest_lr_pow,lowest_lr_pow,total_iterations, high_iterations,n_stage):
    rest = total_iterations - high_iterations
    lr = 10 ** highest_lr_pow * numpy.ones((rest,))
    lr_n = numpy.ceil(numpy.arange(-1,n_stage+1)*(rest/n_stage)).astype(numpy.int32) + high_iterations
    print(lr_n)
    for i in range(n_stage):
        lr[lr_n[i]:lr_n[i+1]] = 10 ** (highest_lr_pow + (lowest_lr_pow-highest_lr_pow)/n_stage*i)
    return lr


def upredict(model, x_batch):
    tensor_x_batch = torch.Tensor(x_batch)
    tensor_x_batch.requires_grad=False
    y = model.forward(tensor_x_batch)
    return y.cpu().detach().numpy()
    
def sn_predict(model, x_batch):
    tensor_x_batch = torch.Tensor(x_batch)
    tensor_x_batch.requires_grad=False
    model.forward(tensor_x_batch)
    return y.cpu().detach().numpy()
    

    
#pts = generate_uniform_points_in_cube(numpy.array([[-1,1],[-1,1]]),400)
#matplotlib.pyplot.plot(pts[:,0],pts[:,1],'.r')
#matplotlib.pyplot.show()
    
#pts = generate_uniform_points_on_cube(numpy.array([[-2,3],[-1,4]]),400)
#matplotlib.pyplot.plot(pts[:,0],pts[:,1],'.r')
#matplotlib.pyplot.show()

#pts1, pts2 = generate_uniform_points_on_cube_time_dependent(numpy.array([[-1,1]]),numpy.array([[0,1]]),20,50)
#matplotlib.pyplot.plot(pts1[:,0],pts1[:,1],'.b')
#matplotlib.pyplot.plot(pts2[:,0],pts2[:,1],'.r')
#matplotlib.pyplot.show()
    
# pts = generate_uniform_points_in_sphere(100,0.1,1000,if_seed=False)
# matplotlib.pyplot.plot(pts[:,0],pts[:,1],'.r')
# matplotlib.pyplot.show()
# print("min |x|=%2.5f, max |x|=%2.5f" %(numpy.min(numpy.sqrt(numpy.sum(pts**2,1))),numpy.max(numpy.sqrt(numpy.sum(pts**2,1)))))


from src.texture_optimization.TextureOptimizer import TextureOptimizationGridSearch


if __name__ == "__main__":
    """
    This runs the grid search experiment for optimizing hyper-parameters. 
    """
    texture_optimizer = TextureOptimizationGridSearch(dataset_path='./my_data/texture_prediction/dataset', 
                                                      time_to_optimize=300,
                                                      param_grid = {
                                                        "learning_rate": [1.0, 0.5, 0.1],
                                                        "loss_rgb": [15000, 30000, 25000],
                                                        "loss_penalization": [0.15, 0.3, 0.5],
                                                        "momentum": [0.9],
                                                      }
                                                    )
    # We will simply run grid search with just one object and one parameter combination. 
    texture_optimizer.grid_search()
    texture_optimizer.compute_best_parameters()
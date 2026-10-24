from grid import Grid
from game.engine import Game, build_patrol_routes, find_spawn_position

grid = Grid.from_ascii("""\
##############################
#S.........#.........#........#
#.........#.........#........#
#.........#.........#........#
#.........#.........#........#
#.........#.........#........#
#.........#.........#........#
#.........#....V....#........#
#.........#.........#........#
#.........#.........#........#
#.........#.........#........#
#.........#.........#........#
#.........#.........#........#
#...........................E#
##############################
""")

print(f'Grid: {grid.width}x{grid.height}')
print(f'Start: {grid.start}')
print(f'Vault: {grid.vault}')
print(f'Exit: {grid.exit}')
print(f'Zone of start: {grid.zone_of(grid.start)}')
print(f'Zone of vault: {grid.zone_of(grid.vault)}')
print(f'Zone of exit: {grid.zone_of(grid.exit)}')

west, middle, east = build_patrol_routes(grid)
print(f'West waypoints: {west}')
print(f'Middle waypoints: {middle}')
print(f'East waypoints: {east}')

# Test spawn positions
used = []
spawn_w = find_spawn_position(grid, min_distance=8, exclude=used, zone='west', avoid_junction_rows=True)
used.append(spawn_w)
spawn_m = find_spawn_position(grid, min_distance=8, exclude=used, zone='middle', avoid_junction_rows=True)
used.append(spawn_m)
spawn_e = find_spawn_position(grid, min_distance=8, exclude=used, zone='east', avoid_junction_rows=True)
print(f'West spawn: {spawn_w} (zone: {grid.zone_of(spawn_w)}, dist: {grid.path_distance(grid.start, spawn_w)})')
print(f'Middle spawn: {spawn_m} (zone: {grid.zone_of(spawn_m)}, dist: {grid.path_distance(grid.start, spawn_m)})')
print(f'East spawn: {spawn_e} (zone: {grid.zone_of(spawn_e)}, dist: {grid.path_distance(grid.start, spawn_e)})')

game = Game(grid)
print(f'Enemy positions: {[(e.position, e.state.value) for e in game.enemies]}')
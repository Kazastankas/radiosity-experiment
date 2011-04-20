/**
 * @file main.hpp
 * @brief OpenGL project application and main function.
 *
 * @author Eric Butler (edbutler)
 */

#include "radiosity.hpp"
#include "application.hpp"
#include "cutil_math.hpp"
#include <SDL/SDL_opengl.h>
#include <stdio.h>

#define PI 3.1415926535f

namespace radiosity {

static const SDLKey KEY_SCREENSHOT = SDLK_f;
static const float MIN_CAM_PHI = PI / 8.0f;
static const float MAX_CAM_PHI = PI;
static const float MIN_CAM_DIST = 1.0f;
static const float CAM_ROTATE_SPEED = 5.0f;
static const float CAM_MOVE_SPEED = 10.0f;

static const int WIDTH = 64;
static const int HEIGHT = 64;
static const double FPS = 60.0;
static const char* TITLE = "Radiosity Renderer";

bool initialize_scene(Scene* scene) 
{
  Light outside_light;
  outside_light.pos = make_float3(0, 0, 10);
  outside_light.color = make_float3(1, 1, 1);
  scene->lights.push_back(outside_light);
  
  Plane top_wall;
  top_wall.corner_pos = make_float3(-5, 5, -5);
  top_wall.color = make_float3(0, 1, 0);
  top_wall.x_vec = make_float3(10, 0, 0);
  top_wall.y_vec = make_float3(0, 0, 10);
  top_wall.x_min = top_wall.y_min = 0;
  top_wall.x_max = top_wall.y_max = 1;
  
  Plane bot_wall;
  bot_wall.corner_pos = make_float3(-5, -5, -5);
  bot_wall.color = make_float3(1, 0, 1);
  bot_wall.x_vec = make_float3(10, 0, 0);
  bot_wall.y_vec = make_float3(0, 0, 10);
  bot_wall.x_min = bot_wall.y_min = 0;
  bot_wall.x_max = bot_wall.y_max = 1;
  
  Plane left_wall;
  left_wall.corner_pos = make_float3(-5, -5, -5);
  left_wall.color = make_float3(0, 1, 1);
  left_wall.x_vec = make_float3(0, 10, 0);
  left_wall.y_vec = make_float3(0, 0, 10);
  left_wall.x_min = left_wall.y_min = 0;
  left_wall.x_max = left_wall.y_max = 1;
  
  Plane right_wall;
  right_wall.corner_pos = make_float3(5, -5, -5);
  right_wall.color = make_float3(1, 0, 0);
  right_wall.x_vec = make_float3(0, 10, 0);
  right_wall.y_vec = make_float3(0, 0, 10);
  right_wall.x_min = right_wall.y_min = 0;
  right_wall.x_max = right_wall.y_max = 1;
  
  Plane front_wall;
  front_wall.corner_pos = make_float3(-5, -5, -5);
  front_wall.color = make_float3(1, 1, 0);
  front_wall.x_vec = make_float3(0, 10, 0);
  front_wall.y_vec = make_float3(10, 0, 0);
  front_wall.x_min = front_wall.y_min = 0;
  front_wall.x_max = front_wall.y_max = 1;
  
  Plane back_wall_1;
  back_wall_1.corner_pos = make_float3(-5, -5, 5);
  back_wall_1.color = make_float3(0, 0, 1);
  back_wall_1.x_vec = make_float3(0, 10, 0);
  back_wall_1.y_vec = make_float3(4, 0, 0);
  back_wall_1.x_min = back_wall_1.y_min = 0;
  back_wall_1.x_max = back_wall_1.y_max = 1;
  
  Plane back_wall_2;
  back_wall_2.corner_pos = make_float3(1, -5, 5);
  back_wall_2.color = make_float3(0, 0, 1);
  back_wall_2.x_vec = make_float3(0, 10, 0);
  back_wall_2.y_vec = make_float3(4, 0, 0);
  back_wall_2.x_min = back_wall_2.y_min = 0;
  back_wall_2.x_max = back_wall_2.y_max = 1;
  
  Plane back_wall_3;
  back_wall_3.corner_pos = make_float3(-1, -5, 5);
  back_wall_3.color = make_float3(0, 0, 1);
  back_wall_3.x_vec = make_float3(0, 4, 0);
  back_wall_3.y_vec = make_float3(2, 0, 0);
  back_wall_3.x_min = back_wall_3.y_min = 0;
  back_wall_3.x_max = back_wall_3.y_max = 1;
   
  Plane back_wall_4;
  back_wall_4.corner_pos = make_float3(-1, 1, 5);
  back_wall_4.color = make_float3(0, 0, 1);
  back_wall_4.x_vec = make_float3(0, 4, 0);
  back_wall_4.y_vec = make_float3(2, 0, 0);
  back_wall_4.x_min = back_wall_4.y_min = 0;
  back_wall_4.x_max = back_wall_4.y_max = 1;
  
  scene->objs.push_back(top_wall);
  scene->objs.push_back(bot_wall);
  scene->objs.push_back(left_wall);
  scene->objs.push_back(right_wall);
  scene->objs.push_back(front_wall);
  scene->objs.push_back(back_wall_1);
  scene->objs.push_back(back_wall_2);
  scene->objs.push_back(back_wall_3);
  scene->objs.push_back(back_wall_4);
  return true;
}

class RadiosityApplication : public Application
{
public:

  RadiosityApplication() { }
  virtual ~RadiosityApplication() { }

  virtual bool initialize();
  virtual void destroy();
  virtual void update( double dt );
  virtual void render();
  virtual void handle_event( const SDL_Event& event );

	enum KeyDir { KD_NEG, KD_ZERO, KD_POS };

	uint8_t* color_data;
  Scene scene_data;
  GLuint texture;

	struct {
		KeyDir horz;
		KeyDir vert;
		KeyDir move_horz;
		KeyDir move_vert;
	} keys;

	struct {
		float fov;
		float aspect;
		float theta;
		float phi;
        float3 position;
	} camera;
};

bool RadiosityApplication::initialize()
{
  bool rv = true;

	color_data = new uint8_t[WIDTH*HEIGHT*4];
	memset( color_data, 0, WIDTH*HEIGHT*4 );

	rv = rv && initialize_scene(&scene_data);
	rv = rv && initialize_radiosity(&scene_data);

  glClearColor( 0, 0, 0, 0 );
  glEnable( GL_BLEND );
  glEnable( GL_TEXTURE_2D );
  glGenTextures( 1, &texture );
  rv = rv && texture > 0;
  glBindTexture( GL_TEXTURE_2D, texture );
  glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  glOrtho( 0, 1, 0, 1, -1, 1 );
  glMatrixMode( GL_MODELVIEW );
  glLoadIdentity();

	camera.theta = 0.0f;
	camera.phi = 3.14f / 6.0f;
	camera.position = make_float3( 0, 0, 0 );

  camera.fov = (PI / 180.0f) * 64.0f;
  camera.aspect = (float)WIDTH / (float)HEIGHT;

	keys.horz = KD_ZERO;
	keys.vert = KD_ZERO;
	keys.move_horz = KD_ZERO;
	keys.move_vert = KD_ZERO;

  return rv;
}

void RadiosityApplication::destroy()
{
	delete [] color_data;
}

void RadiosityApplication::update( double dt )
{
	switch ( keys.horz ) {
	case KD_NEG:
		camera.theta = fmod( camera.theta - CAM_ROTATE_SPEED * dt, 2*PI );
		break;
	case KD_POS:
		camera.theta = fmod( camera.theta + CAM_ROTATE_SPEED * dt, 2*PI );
		break;
	default:
		break;
	}

	switch ( keys.vert ) {
	case KD_NEG:
		camera.phi = fmax( MIN_CAM_PHI, camera.phi - CAM_ROTATE_SPEED * dt );
		break;
	case KD_POS:
		camera.phi = fmin( MAX_CAM_PHI, camera.phi + CAM_ROTATE_SPEED * dt );
		break;
	default:
		break;
	}

  	float3 sidedir = -make_float3( cos(camera.theta + PI/2.0)*sin(camera.phi), cos(camera.phi), sin(camera.theta)*sin(camera.phi) );
	switch ( keys.move_horz ) {
	case KD_NEG:
		camera.position -= CAM_MOVE_SPEED * dt * sidedir;
		break;
	case KD_POS:
		camera.position += CAM_MOVE_SPEED * dt * sidedir;
		break;
	default:
		break;
	}

  	float3 camdir = -make_float3( cos(camera.theta)*sin(camera.phi), cos(camera.phi), sin(camera.theta)*sin(camera.phi) );
	switch ( keys.move_vert ) {
	case KD_NEG:
		camera.position -= CAM_MOVE_SPEED * dt * camdir;
		break;
	case KD_POS:
		camera.position += CAM_MOVE_SPEED * dt * camdir;
		break;
	default:
		break;
	}
}

void RadiosityApplication::render()
{
	Camera cam;

  float3 camdir = -make_float3( cos(camera.theta)*sin(camera.phi), cos(camera.phi), sin(camera.theta)*sin(camera.phi) );
  cam.pos = camera.position;
  // cam.pos = -normalize(camdir) * 2.0f + CAM_POS;
  cam.dir = camdir;
  cam.up = cross( cross( camdir, make_float3( 0, 1, 0 ) ), camdir );
  cam.fov = camera.fov;
  cam.aspect_ratio = camera.aspect;

	render_image(color_data, WIDTH, HEIGHT, &scene_data, &cam);

  glClear( GL_COLOR_BUFFER_BIT );
  glBindTexture( GL_TEXTURE_2D, texture );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, color_data );
  glBegin( GL_QUADS );
  glTexCoord2f( 0.0f, 0.0f );
  glVertex2f( 0.0f, 0.0f );
  glTexCoord2f( 1.0f, 0.0f );
  glVertex2f( 1.0f, 0.0f );
  glTexCoord2f( 1.0f, 1.0f );
  glVertex2f( 1.0f, 1.0f );
  glTexCoord2f( 0.0f, 1.0f );
  glVertex2f( 0.0f, 1.0f );
  glEnd();
}

void RadiosityApplication::handle_event( const SDL_Event& event )
{
  // open to add more event handlers
  switch ( event.type )
  {
  case SDL_KEYDOWN:
      switch ( event.key.keysym.sym )
      {
      case KEY_SCREENSHOT:
          take_screenshot();
          break;
  //Camera look
  case SDLK_LEFT:
    keys.horz = KD_NEG;
    break;
  case SDLK_RIGHT:
    keys.horz = KD_POS;
    break;
  case SDLK_DOWN:
    keys.vert = KD_NEG;
    break;
  case SDLK_UP:
    keys.vert = KD_POS;
    break;
  //Camera move
  case SDLK_c:
  case SDLK_w:
    keys.move_vert = KD_POS;
    break;
  case SDLK_t:
  case SDLK_s:
    keys.move_vert = KD_NEG;
    break;
  case SDLK_h:
  case SDLK_a:
    keys.move_horz = KD_NEG;
    break;
  case SDLK_n:
  case SDLK_d:
    keys.move_horz = KD_POS;
    break;
      default:
          break;
      }
  break;
  case SDL_KEYUP:
      switch ( event.key.keysym.sym )
      {
  //Camera look
  case SDLK_LEFT:
    if ( keys.horz == KD_NEG )
      keys.horz = KD_ZERO;
    break;
  case SDLK_RIGHT:
    if ( keys.horz == KD_POS )
      keys.horz = KD_ZERO;
    break;
  case SDLK_DOWN:
    if ( keys.vert == KD_NEG )
      keys.vert = KD_ZERO;
    break;
  case SDLK_UP:
    if ( keys.vert == KD_POS )
      keys.vert = KD_ZERO;
    break;
  //Camera move
  case SDLK_c:
  case SDLK_w:
    if ( keys.move_vert == KD_POS )
      keys.move_vert = KD_ZERO;
    break;
  case SDLK_t:
  case SDLK_s:
    if ( keys.move_vert == KD_NEG )
      keys.move_vert = KD_ZERO;
    break;
  case SDLK_h:
  case SDLK_a:
    if ( keys.move_horz == KD_NEG )
      keys.move_horz = KD_ZERO;
    break;
  case SDLK_n:
  case SDLK_d:
    if ( keys.move_horz == KD_POS )
      keys.move_horz = KD_ZERO;
    break;
      default:
          break;
      }
  break;
  default:
      break;
  }
}

} /* radiosity */

using namespace radiosity;

static void print_usage( const char* progname )
{
    /*
    printf( "Usage: %s <num_cameras> <image_size> [data_dir]\n\twhere"\
            "\n\tnum_cameras = {545, 2113}\n\t"\
            "image_size = {128, 256}\n\t"\
            "data_dir is the root directory of the lightfield data,\n\t"\
            "or the default /afs location if not given. Do not provide\n\t"\
            "this argument when running on GHC.\n", progname );
    */
}

int main( int argc, char* argv[] )
{
  RadiosityApplication app;
  int rv;

  // start a new application
  if ( argc > 1 ) {
      goto FAIL;
  }

  rv = Application::start_application( &app, WIDTH, HEIGHT, FPS, TITLE );
	cudaThreadExit();
  return 0;

  FAIL:
    print_usage( argv[0] );
    return 1;
}


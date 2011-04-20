/**
 * @file main.hpp
 * @brief OpenGL project application and main function.
 *
 * @author Eric Butler (edbutler)
 */

#include "scene.hpp"
#include "application.hpp"
#include "cutil_math.hpp"
#include <GL/gl.h>
#include <GL/glu.h>
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

static const int WIDTH = 512;
static const int HEIGHT = 512;
static const double FPS = 60.0;
static const char* TITLE = "Radiosity Renderer";

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
  float *rad_matrix;
  size_t matrix_dim;
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
	rv = rv && initialize_radiosity(&scene_data, rad_matrix, &matrix_dim);

  glClearColor( 0, 0, 0, 0 );
  glEnable( GL_BLEND );
  glEnable( GL_TEXTURE_2D );
  glEnable( GL_DEPTH_TEST );
  
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
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
	delete [] rad_matrix;
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
  cam.dir = camdir;
  cam.up = cross( cross( camdir, make_float3( 0, 1, 0 ) ), camdir );
  cam.fov = camera.fov;
  cam.aspect_ratio = camera.aspect;

  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluPerspective(cam.fov * 180.0 / PI,
				 cam.aspect_ratio,
				 0.01f,
				 100.0f);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  //Set camera parameters
  gluLookAt( cam.pos.x, cam.pos.y, cam.pos.z,
             cam.dir.x, cam.dir.y, cam.dir.z,
			       cam.up.x,  cam.up.y,  cam.up.z);

	draw_scene(&scene_data);
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


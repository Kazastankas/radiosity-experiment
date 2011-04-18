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
static const float MAX_CAM_PHI = 7.0f * PI / 16.0f;
static const float MIN_CAM_DIST = 12.0f;
static const float CAM_ROTATE_SPEED = 0.4f;
static const float CAM_ZOOM_SPEED = 10.0f;
static const float3 CAM_POS = make_float3( 0, 2, 0 );

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
  GLuint texture;

	struct {
		KeyDir horz;
		KeyDir vert;
		KeyDir zoom;
	} keys;

	struct {
		float fov;
		float aspect;
		float distance;
		float theta;
		float phi;
	} camera;
};

bool initialize_scene(Scene* scene) 
{
  return true;
}

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
  // glGenTextures( 1, &texture );
  // rv = rv && texture > 0;
  // glBindTexture( GL_TEXTURE_2D, texture );
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

	camera.distance = 80.0f;
	camera.theta = 0.0f;
	camera.phi = 3.14f / 6.0f;

  camera.fov = (PI / 180.0f) * 64.0f;
  camera.aspect = (float)WIDTH / (float)HEIGHT;

	keys.horz = KD_ZERO;
	keys.vert = KD_ZERO;
	keys.zoom = KD_ZERO;

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
		camera.theta = fmod( camera.theta + CAM_ROTATE_SPEED * dt, 2*PI );
		break;
	case KD_POS:
		camera.theta = fmod( camera.theta - CAM_ROTATE_SPEED * dt, 2*PI );
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

	switch ( keys.zoom ) {
	case KD_NEG:
		camera.distance += CAM_ZOOM_SPEED * dt;
		break;
	case KD_POS:
		camera.distance = fmax( MIN_CAM_DIST, camera.distance - CAM_ZOOM_SPEED * dt );
		break;
	default:
		break;
	}
}

void RadiosityApplication::render()
{
	Camera cam;
	float3 camdir = -make_float3( cos(camera.theta)*cos(camera.phi), sin(camera.phi), sin(camera.theta)*cos(camera.phi) );
  cam.position = -camdir * camera.distance + CAM_POS;
  cam.direction = camdir;
  cam.up = cross( cross( camdir, make_float3( 0, 1, 0 ) ), camdir );
  cam.fov = camera.fov;
  cam.aspect_ratio = camera.aspect;

	render( color_data, WIDTH, HEIGHT, &cam );

  glClear( GL_COLOR_BUFFER_BIT );
  // glBindTexture( GL_TEXTURE_2D, texture );
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
  case SDLK_a:
  case SDLK_LEFT:
    keys.horz = KD_NEG;
    break;
  case SDLK_d:
  case SDLK_RIGHT:
    keys.horz = KD_POS;
    break;
  case SDLK_s:
  case SDLK_DOWN:
    keys.zoom = KD_NEG;
    break;
  case SDLK_w:
  case SDLK_UP:
    keys.zoom = KD_POS;
    break;
  case SDLK_q:
    keys.vert = KD_NEG;
    break;
  case SDLK_e:
    keys.vert = KD_POS;
    break;
      default:
          break;
      }
  break;
  case SDL_KEYUP:
      switch ( event.key.keysym.sym )
      {
  case SDLK_a:
  case SDLK_LEFT:
    if ( keys.horz == KD_NEG )
      keys.horz = KD_ZERO;
    break;
  case SDLK_d:
  case SDLK_RIGHT:
    if ( keys.horz == KD_POS )
      keys.horz = KD_ZERO;
    break;
  case SDLK_s:
  case SDLK_DOWN:
    if ( keys.zoom == KD_NEG )
      keys.zoom = KD_ZERO;
    break;
  case SDLK_w:
  case SDLK_UP:
    if ( keys.zoom == KD_POS )
      keys.zoom = KD_ZERO;
    break;
  case SDLK_q:
    if ( keys.vert == KD_NEG )
      keys.vert = KD_ZERO;
    break;
  case SDLK_e:
    if ( keys.vert == KD_POS )
      keys.vert = KD_ZERO;
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


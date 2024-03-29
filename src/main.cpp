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
#include <SDL/SDL.h>
#include <stdio.h>


namespace radiosity {

static const SDLKey KEY_SCREENSHOT = SDLK_f;
static const float MIN_CAM_PHI = 0;
static const float MAX_CAM_PHI = PI;
static const float MIN_CAM_DIST = 1.0f;
static const float CAM_ROTATE_SPEED = 5.0f;
static const float CAM_MOVE_SPEED = 10.0f;
static const float CAM_NEAR_CLIP  = 0.01f;
static const float CAM_FAR_CLIP   = 100.0f;

static const int WIDTH = 512;
static const int HEIGHT = 512;
static const double FPS = 60.0;
static const char* TITLE = "Radiosity Renderer";

static const double RADIOSITY_STEP = 0.5f;

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

  Scene scene_data_0;
  Scene scene_data_1;
  Scene scene_interpolated;

  size_t matrix_dim;
  float3 *rad_matrix_0;
  float3 *rad_matrix_1;
  float3 *interpolated;

  double time;

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

	rv = rv && initialize_scene(&scene_data_0);
	rv = rv && initialize_scene(&scene_data_1);
	rv = rv && initialize_scene(&scene_interpolated);
	matrix_dim = scene_data_0.patches.size();

	rad_matrix_0 = new float3[matrix_dim * matrix_dim];
	rad_matrix_1 = new float3[matrix_dim * matrix_dim];
	interpolated = new float3[matrix_dim * matrix_dim];

int start_time = SDL_GetTicks();
	rv = rv && calc_radiosity(&scene_data_0, rad_matrix_0, matrix_dim);
	memcpy(rad_matrix_1, rad_matrix_0, matrix_dim*matrix_dim*sizeof(float3));
int end_time = SDL_GetTicks();
printf("patches: %d\n", scene_data_0.patches.size());
printf("time: %f\n", (double)(end_time - start_time)/1000.0f);

  time = 0.0f;


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
	delete [] rad_matrix_0;
	delete [] rad_matrix_1;
	delete [] interpolated;
}

void RadiosityApplication::update( double dt )
{
	time += dt;

	//Radiosity data is stale... update
	if(time > RADIOSITY_STEP)
	{
		//swap buffers:
		float3 *m_temp = rad_matrix_1;
		rad_matrix_1 = rad_matrix_0;
		rad_matrix_0 = m_temp;

		Scene s_temp = scene_data_1;
		scene_data_1 = scene_data_0;
		scene_data_0 = s_temp;

		//Update radiosity data
		update_light(&scene_data_0, rad_matrix_0, matrix_dim, RADIOSITY_STEP);
		time = 0;
	}

	//Interpolate radiosity data to get current frame
	float alpha = time / RADIOSITY_STEP;
	for(size_t xx=0; xx<matrix_dim; xx++)
	{
		for(size_t yy=0; yy<matrix_dim; yy++)
		{
			interpolated[xx*matrix_dim + yy] =
				(alpha)   * rad_matrix_0[xx*matrix_dim + yy] +
				(1-alpha) * rad_matrix_1[xx*matrix_dim + yy];
		}

		scene_interpolated.patches[xx].corner_pos =
			(alpha)   * scene_data_0.patches[xx].corner_pos +
			(1-alpha) * scene_data_1.patches[xx].corner_pos;
	}
	update_radiosity(&scene_interpolated, interpolated, matrix_dim);


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

	//Lateral movement:
	float theta2   = camera.theta + PI/2.0;
  	float3 sidedir = -make_float3( cos(theta2), 0, sin(theta2) );
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

	//Forward movement:
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


  //OpenGL rendering:
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  //Set camera parameters
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluPerspective(cam.fov * 180.0 / PI,
				 cam.aspect_ratio,
				 CAM_NEAR_CLIP,
				 CAM_FAR_CLIP);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  float3 dir = cam.pos + cam.dir;
  gluLookAt( cam.pos.x, cam.pos.y, cam.pos.z,
             dir.x,     dir.y,     dir.z,
		     cam.up.x,  cam.up.y,  cam.up.z);

  //Draw every patch in the scene
	draw_scene(&scene_interpolated);
}

void RadiosityApplication::handle_event( const SDL_Event& event )
{
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
    return 1;
}

